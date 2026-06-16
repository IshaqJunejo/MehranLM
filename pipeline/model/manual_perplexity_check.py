import torch
import numpy as np
import os
import json
from typing import List, Tuple
import numpy as np
import torch.nn.functional as F
from model import char_LSTM

def text_to_tensor(text: str) -> torch.Tensor:
    indices = [char_to_idx.get(c, char_to_idx.get('<UNK>', 0)) for c in text]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

@torch.no_grad()
def compute_perplexity_on_sequence(seq_tensor: torch.Tensor, max_prefix_len: int = 256) -> float:
    seq = seq_tensor[0]
    seq_len = len(seq)
    
    if seq_len < 2:
        return float('inf')
    
    losses = []
    step = 1
    
    for i in range(1, seq_len, step):
        start = max(0, i - max_prefix_len)
        prefix = seq[start:i].unsqueeze(0)
        
        output = model(prefix)
        target = seq[i].unsqueeze(0)
        
        loss = F.cross_entropy(output, target)
        losses.append(loss.item())
    
    if not losses:
        return float('inf')
    
    avg_loss = np.mean(losses)
    perplexity = np.exp(avg_loss)
    return perplexity

if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading Char Vocab
    print("Loading the character vocab ...")
    with open("model/vocab.json", "r") as f:
        vocab_data = json.load(f)

    char_to_idx = vocab_data["char_to_idx"]
    idx_to_char = {int(k): v for k, v in vocab_data["idx_to_char"].items()}

    # Model skeleton
    VOCAB_SIZE = len(char_to_idx)
    EMBED_DIM = 64
    HIDDEN_DIM = 128

    model = char_LSTM.Char_LSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM).to(device)

    # Loading model weights
    print("Loading the model weights ...")
    model.load_state_dict(torch.load("model/char_LSTM_checkpoints/epoch_25.pth", map_location=device))

    print("\n\nEnter a statement: ")
    test_input = (str)(input("\t> "))

    tensor = text_to_tensor(test_input)

    perplexity = compute_perplexity_on_sequence(tensor)
    
    print(f"Given input: \"{test_input}\"")
    print(f"Perplexity: {perplexity}")
