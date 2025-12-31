import os
import math
from collections import Counter

def calculate_entropy(filepath: str) -> float:
    corpus = ""
    if filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            corpus += f.read() + "/n"
    
    char_counts = Counter(corpus)
    total_chars = sum(char_counts.values())

    char_probs = {char: count / total_chars for char, count in char_counts.items()}

    entropy = -sum(p * math.log2(p) for p in char_probs.values())

    return entropy

def show_entropy_stats(corpus_dir: str):
    print("Entropy for files in " + corpus_dir)
    for filename in os.listdir(corpus_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(corpus_dir, filename)
            print(filename + " : " + str(calculate_entropy(filepath)) + " bits per char")
    
    print()

if __name__ == "__main__":
    show_entropy_stats("../../Corpus/Cleaned/")
    show_entropy_stats("../../Corpus/Raw/")
    show_entropy_stats("../../Corpus/Raw/Large/")
