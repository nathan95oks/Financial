
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
import os
import time

# -----------------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data_processed"
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

# Hiperparámetros
MAX_LEN = 40
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
N_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 12
CLIP = 1.0
PATIENCE = 3

print(f"Usando dispositivo: {DEVICE}")

# -----------------------------------------------------------------------------
# 1. CARGA DE DATOS Y VOCABULARIO
# -----------------------------------------------------------------------------
print("Cargando datos procesados...")

vocab = pickle.load(open(f"{DATA_DIR}/vocab.pkl", "rb"))
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/val.csv")

PAD_IDX = vocab["<PAD>"]
UNK_IDX = vocab["<UNK>"]

# -----------------------------------------------------------------------------
# 2. DATASET Y ENCODING
# -----------------------------------------------------------------------------
def basic_tokenize(text):
    return str(text).lower().split()

def encode_text(text, vocab):
    tokens = basic_tokenize(text)
    ids = [vocab.get(tok, UNK_IDX) for tok in tokens]
    return torch.tensor(ids, dtype=torch.long)

class FinancialTweetsDataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["text"]
        label = int(self.df.iloc[idx]["label"])
        encoded = encode_text(text, self.vocab)
        return encoded, label

def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = [len(t) for t in texts]
    max_len = min(MAX_LEN, max(lengths))

    padded = []
    for seq in texts:
        seq = seq[:max_len]
        if len(seq) < max_len:
            seq = torch.cat([seq, torch.tensor([PAD_IDX] * (max_len - len(seq)))])
        padded.append(seq)

    padded = torch.stack(padded)
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor([min(l, max_len) for l in lengths], dtype=torch.long)

    return padded, labels, lengths

train_ds = FinancialTweetsDataset(train_df, vocab)
val_ds = FinancialTweetsDataset(val_df, vocab)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

print("Dataloaders listos.")

# -----------------------------------------------------------------------------
# 3. ATENCIÓN + MODELOS RNN (LSTM, GRU, BiLSTM)
# -----------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, rnn_output, mask):
        scores = self.attn(rnn_output).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.sum(rnn_output * attn_weights.unsqueeze(-1), dim=1)
        return context

class RecurrentClassifier(nn.Module):
    def __init__(self, model_type, vocab_size, embed_dim, hidden_dim, out_dim, n_layers, dropout, pad_idx, bidirectional=False, use_attention=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.model_type = model_type
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        if model_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(embed_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0, bidirectional=bidirectional)
        
        self.rnn_out_dim = hidden_dim * (2 if bidirectional else 1)
        
        if use_attention:
            self.attention = AttentionPooling(self.rnn_out_dim)
        else:
            self.attention = None
            
        self.fc = nn.Linear(self.rnn_out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx

    def forward(self, x, lengths=None):
        mask = (x != self.pad_idx).float()
        emb = self.embedding(x)
        
        if self.model_type == 'lstm':
            rnn_out, (h_n, c_n) = self.rnn(emb)
        else:
            rnn_out, h_n = self.rnn(emb)
            
        if self.use_attention:
            context = self.attention(rnn_out, mask)
        else:
            if self.bidirectional:
                h_last = h_n.view(self.n_layers, 2, x.size(0), self.hidden_dim)[-1]
                context = torch.cat([h_last[0], h_last[1]], dim=1)
            else:
                context = h_n[-1]
                
        logits = self.fc(self.dropout(context))
        return logits


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, count = 0, 0, 0

    for xb, yb, lengths in loader:
        xb, yb, lengths = xb.to(DEVICE), yb.to(DEVICE), lengths.to(DEVICE)

        optimizer.zero_grad()
        logits = model(xb, lengths)
        loss = criterion(logits, yb)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        count += xb.size(0)

    return total_loss / count, correct / count

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, count = 0, 0, 0

    with torch.no_grad():
        for xb, yb, lengths in loader:
            xb, yb, lengths = xb.to(DEVICE), yb.to(DEVICE), lengths.to(DEVICE)
            logits = model(xb, lengths)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            count += xb.size(0)

    return total_loss / count, correct / count

# -----------------------------------------------------------------------------
# 5. PÉRDIDA PONDERADA
# -----------------------------------------------------------------------------
counts = train_df["label"].value_counts().sort_index()
weights = 1.0 / counts
weights = weights / weights.sum()
class_weights = torch.tensor(weights.values, dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# -----------------------------------------------------------------------------
# 6. ENTRENAMIENTO DE MODELOS (LSTM, GRU, BiLSTM)
# -----------------------------------------------------------------------------
histories = {}

configs = [
    {'name': 'lstm_base', 'type': 'lstm', 'bi': False, 'attn': False},
    {'name': 'lstm_attn', 'type': 'lstm', 'bi': False, 'attn': True},
    {'name': 'gru_base',  'type': 'gru',  'bi': False, 'attn': False},
    {'name': 'gru_attn',  'type': 'gru',  'bi': False, 'attn': True},
    {'name': 'lstm_bi',   'type': 'lstm', 'bi': True,  'attn': False},
    {'name': 'lstm_bi_attn','type':'lstm','bi': True,  'attn': True}
]

for conf in configs:
    m_name = conf['name']
    m_type = conf['type']
    bi = conf['bi']
    attn = conf['attn']
    
    print(f"\n{'='*40}")
    print(f" Entrenando: {m_name.upper()} (Bi={bi}, Attn={attn})")
    print(f"{'='*40}")
    
    model = RecurrentClassifier(
        model_type=m_type,
        vocab_size=len(vocab),
        embed_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        out_dim=3,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        pad_idx=vocab["<PAD>"],
        bidirectional=bi,
        use_attention=attn
    ).to(DEVICE)
    
    save_path = f"{MODELS_DIR}/{m_name}_best_model.pth"
    
    # SIEMPRE ENTRENAMOS PARA CAPTURAR TIEMPO
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    start_time = time.time()
    
    for ep in range(EPOCHS):
        tl, ta = train_epoch(model, train_loader, optimizer, criterion)
        vl, va = evaluate(model, val_loader, criterion)
        scheduler.step()
        
        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['train_acc'].append(ta)
        history['val_acc'].append(va)
        
        print(f"Epoch {ep+1}/{EPOCHS} | Train Loss: {tl:.4f} Acc: {ta:.4f} | Val Loss: {vl:.4f} Acc: {va:.4f}")
        
        if va > best_acc:
            best_acc = va
            state = {
                'model_state': model.state_dict(),
                'config': {
                    'model_type': m_type,
                    'vocab_size': len(vocab),
                    'embed_dim': EMBEDDING_DIM, 
                    'hidden_dim': HIDDEN_DIM,
                    'n_layers': N_LAYERS,
                    'dropout': DROPOUT,
                    'pad_idx': vocab["<PAD>"],
                    'bidirectional': bi,
                    'use_attention': attn
                },
                'vocab': vocab
            }
            torch.save(state, save_path)
            print(f" --> Nuevo récord! Modelo guardado en {save_path}")
            
    end_time = time.time()
    history['total_time'] = end_time - start_time
    histories[m_name] = history
    print(f"Tiempo total para {m_name}: {history['total_time']:.2f}s")


# Guardar historial completo para Notebook 4
with open(f"{MODELS_DIR}/histories.pkl", "wb") as f:
    pickle.dump(histories, f)
print(f"\nHistorial guardado en {MODELS_DIR}/histories.pkl")
print("\n¡Proceso finalizado para LSTM, GRU y BiLSTM!")
