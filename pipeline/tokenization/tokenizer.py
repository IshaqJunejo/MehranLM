import os
import json
from typing import Dict, Tuple, List
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Got this number from an EDA script 
NUM_OF_CHAR = 223         # Source: pipeline/EDA/unique_chars.txt

NL_TOKEN = "\uE000"
EW_TOKEN = "\uE001"
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"
MASK_TOKEN = "<MASK>"

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
    corpus = corpus.replace("\n", NL_TOKEN)

    # Tokenization
    tokenizer = Tokenizer(
        BPE(
            unk_token=UNK_TOKEN
            # end_of_word_suffix=EW_TOKEN
        )
    )
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=num_merges + NUM_OF_CHAR,
        min_frequency=1,
        special_tokens=[UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN, MASK_TOKEN],
        end_of_word_suffix=EW_TOKEN
    )

    tokenizer.train_from_iterator(
        corpus_iterator(corpus),
        trainer=trainer
    )

    tokenizer.save("tokenization/tokenizer.json")
    print("Tokenizer saved to \"tokenizer.json\" file")

def encode(text: str, tokenizer) -> List[int]:
    text = text.replace("\n", NL_TOKEN)
    return tokenizer.encode(text).ids

def decode(ids: List[int], tokenizer, show_markers = False) -> str:
    decoded_text = tokenizer.decode(ids)

    if not show_markers:
        return decoded_text.replace(PAD_TOKEN, "").replace(NL_TOKEN, "\n").replace(" ", "").replace(EW_TOKEN, " ")
    elif show_markers:
        return decoded_text

if __name__ == "__main__":
    # # Loading the corpus
    # corpus_dir = "../Corpus/Cleaned/"
    # corpus = ""
    # for filename in os.listdir(corpus_dir):
    #     if filename.endswith(".txt"):
    #         filepath = os.path.join(corpus_dir, filename)
    #         with open(filepath, "r", encoding="utf-8") as f:
    #             corpus += f.read() + "\n"

    # NUM_OF_MERGES = 32000
    # train_bpe(corpus, NUM_OF_MERGES)

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
