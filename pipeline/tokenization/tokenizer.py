import os
import json
from typing import Dict, Tuple, List
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Got this number from an EDA script 
NUM_OF_CHAR = 146 - 5         # total 146 unique characters, and 5 whitespaces

# Load tokenizer dictionary from JSON
def load_tokens(filepath: str):
    return Tokenizer.from_file(filepath)

# Corpus Iterator
def corpus_iterator(corpus: str):
    for line in corpus.splitlines():
        if line.strip():
            yield line.strip()

def train_bpe(corpus: str, num_merges: int):
    # Adding the marker for NEW LINE and END OF WORD
    corpus = corpus.replace("\n", "NL")
    preprocessed_corpus = " ".join([word + "EW" for word in corpus.split()])

    # Tokenization
    tokenizer = Tokenizer( BPE( unk_token="<UNK>" ) )
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=num_merges + NUM_OF_CHAR,
        min_frequency=1,
        special_tokens=["<UNK>", "<PAD>", "<BOS>", "<EOS>", "<SEP>", "<MASK>"]
    )

    tokenizer.train_from_iterator(
        corpus_iterator(preprocessed_corpus),
        trainer=trainer
    )

    tokenizer.save("tokenization/tokenizer.json")
    print("Tokenizer saved to \"tokenizer.json\" file")

def encode(text: str, tokenizer) -> List[int]:
    text = text.replace("\n", "NL")
    preprocessed_text = " ".join([word + "EW" for word in text.split()])
    return tokenizer.encode(preprocessed_text).ids

def decode(ids: List[int], tokenizer, show_markers = False) -> str:
    decoded_text = tokenizer.decode(ids)

    if not show_markers:
        return decoded_text.replace("<PAD>", "").replace(" ", "").replace("EW", " ").replace("NL", "\n")
    elif show_markers:
        return decoded_text.replace(" ", "")

if __name__ == "__main__":
    # # Loading the corpus
    # corpus_dir = "../Corpus/Cleaned/"
    # corpus = ""
    # for filename in os.listdir(corpus_dir):
    #     if filename.endswith(".txt"):
    #         filepath = os.path.join(corpus_dir, filename)
    #         with open(filepath, "r", encoding="utf-8") as f:
    #             corpus += f.read() + "\n"

    # train_bpe(corpus, 8000)

    # Loading the tokenizer
    tokenizer = Tokenizer.from_file("tokenization/tokenizer.json")

    # Shout-out to " حسنين سمون ۽ بابار منگي "
    sample = "منهنجي دل کي رجهاءڻ لاء، رڳو مون سان ڪوڙ هياءي.\nوڏيون وڏيون ڳالهيون ڪياءي، وفا ته ڪانه ڪياءي." 
    print("Our sample text before tokenization")
    print("\n" + sample + "\n\n")

    print("Showing the tokens individually")

    ids = encode(sample, tokenizer)

    for i in range(len(ids)):
        print(decode([ids[i]], tokenizer, True))
    
    print("\nToken List")
    print(ids)
    
    print("\nReconstructed from Tokens")
    print(decode(ids, tokenizer))
