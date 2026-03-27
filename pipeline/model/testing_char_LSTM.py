import torch
import numpy as np
import os
import json
from model import char_LSTM

# The results running this script have been piped to `text_generated_by_char_LSTM.txt` file

if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("model/vocab.json", "r") as f:
        vocab_data = json.load(f)

    char_to_idx = vocab_data["char_to_idx"]
    # idx_to_char = vocab_data["idx_to_char"]
    idx_to_char = {int(k): v for k, v in vocab_data["idx_to_char"].items()}
    
    VOCAB_SIZE = len(char_to_idx)
    EMBED_DIM = 64
    HIDDEN_DIM = 128

    model = char_LSTM.Char_LSTM(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM).to(device)

    model.load_state_dict(torch.load("model/char_LSTM_checkpoints/epoch_25.pth", map_location=device))

    seed = "مھراڻ يونيورسٽي"
    length = 256

    print(f"Seed text:\n{seed}")
    print(f"Generate {length} more characters to it\n")

    for i in range(2, 20, 2):
        temp = i / 10
        print(f"Temperature: {temp}\n")
        sample = char_LSTM.generate(model, seed, length, temp, char_to_idx, idx_to_char)

        print(f"{sample}\n")