import os
import json
from typing import Dict, Tuple, List
import BPE_merges

# Applying our merges to the entire corpus to extract our tokens
def build_vocab_from_corpus(corpus: str, merge_ranks: Dict[Tuple[str, str], int]) -> Dict[str, str]:
    token_set = set()
    tokens = BPE_merges.apply_bpe_to_text(merge_ranks, corpus)
    token_set.update(tokens)

    special_tokens = ["<UNK>", "<PAD>"]
    vocab_dict = {}

    for idx, token in enumerate(special_tokens):
        vocab_dict[token] = idx

    for idx, token in enumerate((sorted(token_set))):
        vocab_dict[token] = idx + len(special_tokens)
    
    # vocab_dict = {token: idx for idx, token in enumerate(sorted(token_set))}
    return vocab_dict

# Write content to a json file
def write_to_json(content: Dict[str, str], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)
    print(f"Saved content to {path}")

# Read content from a json file
def read_from_json(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    return content

# Convert text into IDs
def encode(text: str, token_to_id: Dict[str, str], merge_ranks: Dict[Tuple[str, str], int]) -> List[int]:
    # Convert text into tokens
    tokens = BPE_merges.apply_bpe_to_text(merge_ranks, text)
    # Convert tokens in IDs
    return [token_to_id.get(tok, token_to_id["<UNK>"]) for tok in tokens]

# Convert IDs into text
def decode(ids: List[int], id_to_token: Dict[str, str]) -> str:
    # Convert IDs into tokens
    tokens = [id_to_token.get(str(i)) for i in ids]
    # Convert tokens into text
    return "".join(tokens).replace("</w>", " ").replace("<nl>", "\n")

if __name__ == "__main__":
    # # Loading the corpus
    # corpus_dir = "../../Corpus/Cleaned/"
    # corpus = ""
    # for filename in os.listdir(corpus_dir):
    #     if filename.endswith(".txt"):
    #         filepath = os.path.join(corpus_dir, filename)
    #         with open(filepath, "r", encoding="utf-8") as f:
    #             corpus += f.read() + "\n"

    # Loading merges and merge-priorities
    merges = BPE_merges.load_merges("./merges.txt")
    merge_ranks = BPE_merges.build_merge_ranks(merges)

    # # token_to_id and id_to_token
    # token_vocab = build_vocab_from_corpus(corpus, merge_ranks)
    # id_to_token = {idx: token for token, idx in token_vocab.items()}

    # # saving our token vocab to a file
    # write_to_json(token_vocab, "token_to_id.json")
    # write_to_json(id_to_token, "id_to_token.json")

    token_to_id = read_from_json("token_to_id.json")
    id_to_token = read_from_json("id_to_token.json")

    # Shout-out to " حسنين سمون ۽ بابار منگي "
    sample_text = "منهنجي دل کي رجهاءڻ لاء، رڳو مون سان ڪوڙ هياءي.\nوڏيون وڏيون ڳالهيون ڪياءي، وفا ته ڪانه ڪياءي." 
    print(sample_text)
    print()

    sample_tokens = encode(sample_text, token_to_id, merge_ranks)
    print("Token List")
    print(sample_tokens)
    print()

    reconstructed_text = decode(sample_tokens, id_to_token)
    print("Reconstructed from Tokens")
    print(reconstructed_text)
