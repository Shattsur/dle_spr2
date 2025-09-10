#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing
import evaluate as eval_metric

# -----------------------------
# Настройки
# -----------------------------
BATCH_SIZE = 512
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 1
EPOCHS = 7
LR = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Загрузка данных
# -----------------------------
# df_clean должен содержать колонку 'text'
# df_clean = pd.read_csv("data/texts.csv")  # пример загрузки
# Мини-EDA
df_clean["len_tokens"] = df_clean["text"].apply(lambda x: len(str(x).split()))
MAX_LENGTH = int(np.percentile(df_clean["len_tokens"], 90)) + 10

# -----------------------------
# Токенизация
# -----------------------------
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Whitespace(),
    pre_tokenizers.Punctuation(),
    pre_tokenizers.Digits(individual_digits=True)
])
trainer = trainers.BpeTrainer(
    vocab_size=30000,
    special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
    min_frequency=2
)
tokenizer.train_from_iterator(df_clean["text"].apply(str), trainer=trainer)
tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    special_tokens=[("<s>", 2), ("</s>", 3)]
)
tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
tokenizer.enable_truncation(max_length=MAX_LENGTH)
tokenizer.save("data/tokenizer.json")

# -----------------------------
# Подготовка данных
# -----------------------------
def encode_texts(texts):
    return [tokenizer.encode(text).ids for text in texts]

encoded_texts = encode_texts(df_clean["text"])
X = [seq[:-1] for seq in encoded_texts]
Y = [seq[1:] for seq in encoded_texts]

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

class TextDataset(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

def collate_fn(batch):
    X_batch, Y_batch = zip(*batch)
    lengths = torch.tensor([len(x) for x in X_batch], dtype=torch.long)
    X_padded = pad_sequence([torch.tensor(x) for x in X_batch], batch_first=True, padding_value=0)
    Y_padded = pad_sequence([torch.tensor(y) for y in Y_batch], batch_first=True, padding_value=0)
    return X_padded, Y_padded, lengths

train_loader = DataLoader(TextDataset(X_train, Y_train), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(TextDataset(X_val, Y_val), batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_loader  = DataLoader(TextDataset(X_test, Y_test), batch_size=BATCH_SIZE, collate_fn=collate_fn)

# -----------------------------
# Модель LSTM
# -----------------------------
class ImprovedTextLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.4)
        nn.init.normal_(self.embedding.weight, 0.0, 0.02)
        nn.init.constant_(self.embedding.weight[0], 0)

        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=0.3, bidirectional=False
        )
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, vocab_size)
        )
        self.dropout = nn.Dropout(0.6)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name: nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name: nn.init.orthogonal_(param.data)
            elif 'bias' in name: nn.init.constant_(param.data, 0)

    def forward(self, x, lengths=None, hidden=None):
        x = self.embedding(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.embedding_dropout(x)
        x = self.ln1(x)
        x = self.dropout(x)
        if lengths is not None:
            packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, hidden = self.lstm(packed_x, hidden)
            out, _ = pad_packed_sequence(packed_out, batch_first=True, padding_value=0)
        else:
            out, hidden = self.lstm(x, hidden)
        out = self.ln2(out)
        out = self.dropout(out)
        return self.fc(out), hidden

vocab_size = tokenizer.get_vocab_size()
model_lstm = ImprovedTextLSTM(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.01)
optimizer = optim.AdamW(model_lstm.parameters(), lr=LR, weight_decay=0.05)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)
scaler = torch.amp.GradScaler("cuda")
writer = SummaryWriter(f'runs/text_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

# -----------------------------
# Обучение
# -----------------------------
def train_epoch(model, loader, optimizer, criterion, device, epoch, scaler):
    model.train()
    total_loss, total_tokens = 0, 0
    for X_batch, Y_batch, lengths in tqdm(loader, desc=f"Training Epoch {epoch+1}"):
        X_batch, Y_batch, lengths = X_batch.to(device), Y_batch.to(device), lengths.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            output, _ = model(X_batch, lengths)
            loss = criterion(output.view(-1, output.shape[-1]), Y_batch.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * (Y_batch != 0).sum().item()
        total_tokens += (Y_batch != 0).sum().item()
    return total_loss / total_tokens

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for X_batch, Y_batch, lengths in tqdm(loader, desc="Validation"):
            X_batch, Y_batch, lengths = X_batch.to(device), Y_batch.to(device), lengths.to(device)
            output, _ = model(X_batch, lengths)
            loss = criterion(output.view(-1, output.shape[-1]), Y_batch.view(-1))
            total_loss += loss.item() * (Y_batch != 0).sum().item()
            total_tokens += (Y_batch != 0).sum().item()
    return total_loss / total_tokens

def calculate_rouge_evaluate(model, loader, tokenizer, device, num_samples=500):
    rouge = eval_metric.load("rouge")
    hypotheses, references = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, Y_batch, lengths in tqdm(loader, desc="ROUGE Eval"):
            X_batch, Y_batch, lengths = X_batch.to(device), Y_batch.to(device), lengths.to(device)
            output, _ = model(X_batch, lengths)
            preds = output.argmax(dim=-1)
            for i in range(X_batch.size(0)):
                if len(hypotheses) >= num_samples: break
                pred_tokens = [t for t in preds[i].cpu().tolist() if t not in [0, 2, 3]]
                target_tokens = [t for t in Y_batch[i].cpu().tolist() if t not in [0, 2, 3]]
                pred_text, target_text = tokenizer.decode(pred_tokens), tokenizer.decode(target_tokens)
                if pred_text.strip() and target_text.strip():
                    hypotheses.append(pred_text)
                    references.append(target_text)
    return rouge.compute(predictions=hypotheses, references=references)

best_val_loss = float('inf')
for epoch in range(EPOCHS):
    train_loss = train_epoch(model_lstm, train_loader, optimizer, criterion, DEVICE, epoch, scaler)
    val_loss = evaluate(model_lstm, val_loader, criterion, DEVICE)
    train_ppl, val_ppl = math.exp(train_loss), math.exp(val_loss)
    rouge_val = calculate_rouge_evaluate(model_lstm, val_loader, tokenizer, DEVICE, num_samples=200)

    print(f"Epoch {epoch+1} | Train Loss={train_loss:.4f} PPL={train_ppl:.2f} | Val Loss={val_loss:.4f} PPL={val_ppl:.2f}")
    print("ROUGE:", ", ".join([f"{k}={v:.4f}" for k,v in rouge_val.items()]))

    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Perplexity/train', train_ppl, epoch)
    writer.add_scalar('Perplexity/val', val_ppl, epoch)
    for metric_name, score in rouge_val.items():
        writer.add_scalar(f'ROUGE/{metric_name}', score, epoch)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model_lstm.state_dict(), "data/best_model_lstm.pt")

writer.close()

# -----------------------------
# Генерация текста
# -----------------------------
def generate_text_topp(model, tokenizer, prompt, max_length=MAX_LENGTH, temperature=0.7, top_p=0.9):
    model.eval()
    input_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(DEVICE)
    generated_tokens = []
    with torch.no_grad():
        for _ in range(max_length):
            output, _ = model(input_tensor)
            logits = output[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_keep = cumulative_probs <= top_p
            sorted_indices_to_keep[..., 0] = True
            filtered_probs = torch.zeros_like(probs)
            filtered_probs[0, sorted_indices[0, sorted_indices_to_keep[0]]] = sorted_probs[0, sorted_indices_to_keep[0]]
            filtered_probs /= filtered_probs.sum()
            next_token = torch.multinomial(filtered_probs, 1)
            if next_token.item() == tokenizer.token_to_id("</s>"): break
            generated_tokens.append(next_token.item())
            input_tensor = torch.cat([input_tensor, next_token], dim=-1)
    return tokenizer.decode(generated_tokens)

# Примеры генерации
test_prompts = ["I love", "The weather is", "What do you think about"]
for prompt in test_prompts:
    print(f"{prompt} -> {prompt} {generate_text_topp(model_lstm, tokenizer, prompt)}")

# -----------------------------
# Perplexity на тесте
# -----------------------------
def calculate_perplexity(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for X_batch, Y_batch, lengths in loader:
            X_batch, Y_batch, lengths = X_batch.to(device), Y_batch.to(device), lengths.to(device)
            output, _ = model(X_batch, lengths)
            loss = F.cross_entropy(output.view(-1, output.shape[-1]), Y_batch.view(-1), reduction='sum', ignore_index=0)
            total_loss += loss.item()
            total_tokens += (Y_batch != 0).sum().item()
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)

print(f"Perplexity на тестовой выборке: {calculate_perplexity(model_lstm, test_loader, DEVICE):.4f}")
