import torch
import numpy as np
import os
import json
from typing import List, Tuple
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from model import char_LSTM

def text_to_tensor(text: str) -> torch.Tensor:
    indices = [char_to_idx.get(c, char_to_idx.get('<UNK>', 0)) for c in text]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

def chunk_text(text: str, chunk_size: int) -> List[str]:
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

@torch.no_grad()
def compute_perplexity_on_sequence(seq_tensor: torch.Tensor, max_prefix_len: int = 256) -> float:
    seq = seq_tensor[0]
    seq_len = len(seq)
    
    if seq_len < 2:
        return float('inf')
    
    losses = []
    step = 8
    
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

def compute_perplexity_on_file(path: str, filename: str):
    file_contents = ""

    if filename.endswith(".txt"):
        filepath = os.path.join(path, filename)

        print(f"File name: {filename}")
        with open(filepath, "r", encoding="utf-8") as f:
            file_contents = f.read()
    else:
        return
    
    print(f"File size: {len(file_contents)} characters")

    chunks = chunk_text(file_contents, 512)
    perplexities = []

    print(f"Computing perplexities on file {len(chunks)} chunks ...")
    for chunk in tqdm(chunks):
        if len(chunk) < 10:
            continue
        tensor = text_to_tensor(chunk)
        ppl = compute_perplexity_on_sequence(tensor)
        if np.isfinite(ppl):
            perplexities.append(ppl)
    
    avg_ppl = np.mean(perplexities)
    median_ppl = np.median(perplexities)
    std_ppl = np.std(perplexities)

    print(f"Average Perplexity: {avg_ppl}")
    print(f"Median Perplexity: {median_ppl}")
    print(f"Standard Deviation: {std_ppl}")

    output = {
        "perplexities": perplexities,
        "average": avg_ppl,
        "median": median_ppl,
        "standard deviation": std_ppl
    }

    # output_dir = "model/perplexity_scores/Private/"
    output_dir = "model/perplexity_scores/"
    output_filename = filename.replace(".txt", "_results.json")

    output_filepath = os.path.join(output_dir, output_filename)
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    
    print(f"Perplexity scores written to the file {output_filename}\n\n")

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

    # Calculating Perplexity
    # corpus_dir = "../Corpus/Private/Cleaned/"
    corpus_dir = "../Corpus/Cleaned/"

    for filename in os.listdir(corpus_dir):
        if filename.endswith(".txt"):
            compute_perplexity_on_file(corpus_dir, filename)
    