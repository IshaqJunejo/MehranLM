from collections import Counter, defaultdict
import os
import re
import json
from typing import Dict, Tuple, List

def get_vocab_from_corpus(corpus: str) -> Counter:
    vocab = Counter()

    corpus_lines = corpus.splitlines()

    for line in corpus_lines:
        for word in line.split():
            symbols = tuple(list(word) + ["</w>"])
            vocab[symbols] += 1

    return vocab

def get_pair_frequency(vocab: Counter) -> Dict[Tuple[str, str], int]:
    pairs = defaultdict(int)

    for words, freq in vocab.items():
        for i in range(len(words) - 1):
            pair = (words[i], words[i + 1])
            pairs[pair] += freq
    
    return pairs

def merge_pairs_in_vocab(pair: Tuple[str, str], vocab: Counter) -> Counter:
    merged_vocab = Counter()
    a, b = pair
    pattern = re.escape(a) + r' ' + re.escape(b) # same as `pattern = a + ' ' + b` just with regex safety
    for words, freq in vocab.items():
        s = " ".join(words)
        s_new = re.sub(pattern, a + b, s)

        new_words = tuple(s_new.split(" "))
        merged_vocab[new_words] += freq
    
    return merged_vocab

def train_bpe(corpus: str, num_merges: int) -> List[Tuple[str, str]]:
    vocab = get_vocab_from_corpus(corpus)
    merges: List[Tuple[str, str]] = []

    for i in range(num_merges):
        pairs = get_pair_frequency(vocab)

        if not pairs:
            break

        best_pair, best_count = max(pairs.items(), key=lambda kv : kv[1])

        if best_count < 1:
            break

        merges.append(best_pair)
        vocab = merge_pairs_in_vocab(best_pair, vocab)
    
    return merges

def apply_bpe_to_word(merge_ranks: Dict[Tuple[str, str], int], word: str) -> List[str]:
    symbols = list(word) + ["</w>"]

    while True:
        pairs = ((symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1))
        candidate_ranks = []

        for idx, pair in enumerate(pairs):
            if pair in merge_ranks:
                candidate_ranks.append((merge_ranks[pair], pair, idx))
        if not candidate_ranks:
            break

        candidate_ranks.sort()
        _, pair_to_merge, idx_to_merge = candidate_ranks[0]
        a, b = pair_to_merge

        symbols = symbols[:idx_to_merge] + [a + b] + symbols[idx_to_merge + 2:]

    return symbols


def apply_bpe_to_text(merge_ranks: Dict[Tuple[str, str], int], text: str) -> List[List[str]]:
    text_lines = text.splitlines()
    out_lines = []

    for line in text_lines:
        line_tokens = []
        for word in line.strip().split():
            bpe_on_word = apply_bpe_to_word(merge_ranks, word)
            line_tokens.extend(bpe_on_word)
        out_lines.append(line_tokens)
    
    return out_lines


def build_merge_ranks(merges: List[Tuple[str, str]]) -> Dict[Tuple[str, str], int]:
    return {pair: idx for idx, pair in enumerate(merges)}

def save_merges(merges: List[Tuple[str, str]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a} {b}\n")

def load_merges(path: str) -> List[Tuple[str, str]]:
    merges = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                merges.append((parts[0], parts[1]))
    
    return merges

if __name__ == "__main__":
    # input_dir = "../../Corpus/Cleaned/"

    # corpus = ""

    # for filename in os.listdir(input_dir):
    #     if filename.endswith(".txt"):
    #         filepath = os.path.join(input_dir, filename)
    #         with open(filepath, "r", encoding="utf-8") as f:
    #             corpus += f.read() + "\n"
    
    # vocab = get_vocab_from_corpus(corpus)

    # merges = train_bpe(corpus, 5000)

    # save_merges(merges, "./merges.txt")
    # print("Saved merges to \'merges.txt\'")

    # print(vocab)

    merges = load_merges("./merges.txt")
    merge_ranks = build_merge_ranks(merges)

    # print(merges)
    sample = "منهنجي دل کي رجهاءڻ لاء قسم به ڪوڙا کيان.\nوڏيون وڏيون ڳالهين ڪياءي."
    print("Our sample text before tokenization")
    print("\n" + sample + "\n\n")

    sample_tokens = apply_bpe_to_text(merge_ranks, sample)

    print("After tokenization")

    for i in sample_tokens:
        for j in i:
            print(j)
    