import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CharDataset(Dataset):
    def __init__(self, text, seq_length, char_to_idx):
        self.text = text
        self.seq_length = seq_length
        self.char_to_idx = char_to_idx
    
    def __len__(self):
        return len(self.text) - self.seq_length
    
    def __getitem__(self, idx):
        # Window of input chars
        chunk = self.text[idx : idx + self.seq_length]
        # Target char (right after the input chars)
        target = self.text[idx + self.seq_length]

        return (
            torch.tensor([self.char_to_idx[c] for c in chunk], dtype=torch.long),
            torch.tensor([self.char_to_idx[target]], dtype=torch.long)
        )

class Char_LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(Char_LSTM, self).__init__()

        # Model
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (h_n, _) = self.lstm(embedded)
        hidden = h_n[-1, :, :]
        output = self.fc(self.dropout(hidden))

        return output

def train_model(model, dataloader, char_to_idx, idx_to_char, epochs, batch_size, lr, val_data=None, seed=None):
    print(f"Using device: {device}")
    model = model.to(device)

    if seed != None:
        print(f"The model will be tested for text generation on the following seed text:\n{seed}\n")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            # Calculate Loss
            loss = criterion(output, batch_y.squeeze())
            # Backprop
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)

        print(f"Epoch: {epoch + 1}/{epochs}  |  Training Loss: {avg_loss:.4f}", end="")

        if val_data != None:
            model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch_idx, (batch_X, batch_y) in enumerate(val_data):
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    output = model(batch_X)
                    loss = criterion(output, batch_y.squeeze())
                    val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_data)
                val_losses.append(avg_val_loss)

            print(f"  |  Validation Loss: {avg_val_loss:.4f}")
        else:
            print("")

        if epoch % 5 == 1 and seed != None:
            model.eval()
            print("\n--- Text Generation Test ---")

            sample = generate(model, seed, 100, 0.8, char_to_idx, idx_to_char)

            print(f"Sample generated:\n{sample}\n")
            model.train()

        torch.save(model.state_dict(), f"./model/char_LSTM_checkpoints/epoch_{epoch+1}.pth")


def generate(model, seed_string, length, temperature, char_to_idx, idx_to_char):
    model.eval()

    input_idx = [char_to_idx[c] for c in seed_string]
    generated_text = seed_string

    with torch.no_grad():
        for i in range(length):
            x = torch.tensor([input_idx], dtype=torch.long).to(device)

            output = model(x)

            logits = output.squeeze() / max(temperature, 1e-6)
            probabilities = F.softmax(logits, dim=0)

            char_idx = torch.multinomial(probabilities, 1).item()

            char = idx_to_char[char_idx]
            generated_text += char

            input_idx.append(char_idx)
    
    return generated_text

if __name__ == "__main__":
    input_dir = "../Corpus/subcorpus-for-charLM/"
    corpus = ""

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                corpus += f.read() + "\n\n"
    
    all_chars = sorted(list(set(corpus)))

    char_to_idx = {ch: i for i, ch in enumerate(all_chars)}
    idx_to_char = {i: ch for i, ch in enumerate(all_chars)}
    
    #
    train_size = int(0.8 * len(corpus))
    val_size = int(0.1 * len(corpus))

    train_corpus = corpus[:train_size]
    val_corpus = corpus[train_size : train_size + val_size]
    test_corpus = corpus[train_size + val_size :]

    #
    SEQ_LEN = 100
    BATCH_SIZE = 64

    train_ds = CharDataset(train_corpus, SEQ_LEN, char_to_idx)
    val_ds = CharDataset(val_corpus, SEQ_LEN, char_to_idx)
    test_ds = CharDataset(test_corpus, SEQ_LEN, char_to_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    VOCAB_SIZE = len(all_chars)
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    LR = 0.001
    EPOCHS = 25

    model = Char_LSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM).to(device)

    seed = "ڪمپيوٽر سائنس جي دنيا ۾ سڀ کان اهم"

    train_model(model, train_loader, char_to_idx, idx_to_char, EPOCHS, BATCH_SIZE, LR, val_loader, seed)

    criterion = nn.CrossEntropyLoss()

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (batch_X, batch_y) in enumerate(test_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            output = model(batch_X)
            loss = criterion(output, batch_y.squeeze())

            test_loss += loss
    
    print(f"Testing Loss: {test_loss / len(test_loader):.4f}")

    with open("model/vocab.json", "w") as f:
        json.dump({"char_to_idx": char_to_idx, "idx_to_char": idx_to_char}, f)
