import os
from typing import List

from tokenizers import Tokenizer, normalizers, decoders, AddedToken, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFC, StripAccents, Replace, Sequence as NormalizerSequence
from tokenizers.decoders import ByteFallback, BPEDecoder, Sequence as DecoderSequence

from transformers import PreTrainedTokenizerFast, AutoTokenizer

# Got this number from an EDA script
NUM_OF_CHAR = 232         # Source: pipeline/EDA/unique_chars.txt

NL_TOKEN = "\uE000"
EW_TOKEN = "\uE001"
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"
MASK_TOKEN = "<MASK>"

# Byte Fallback
BYTE_FALLBACK_TOKENS = [f"<0x{i:02X}>" for i in range(256)]


def load_tokens(filepath: str):
    return Tokenizer.from_file(filepath)


def corpus_iterator(corpus: str):
    for line in corpus.split("\n"):
        stripped = line.strip()
        if stripped:
            yield stripped + "\n"
 

def train_bpe(corpus: str, num_merges: int, save_dir: str = "tokenization/files"):
    tokenizer = Tokenizer(
        BPE(
            unk_token=UNK_TOKEN,
            byte_fallback=True,
        )
    )

    tokenizer.add_special_tokens([
        UNK_TOKEN, 
        AddedToken(NL_TOKEN, normalized=True, single_word=False), 
        PAD_TOKEN, 
        BOS_TOKEN, 
        EOS_TOKEN, 
        SEP_TOKEN, 
        MASK_TOKEN
    ])

    DIACRITICS_REGEX = r"[\u064B\u064C\u064D\u064E\u065E\u064F\u0650\u0651\u0652]"

    # Normalizer Route
    tokenizer.normalizer = NormalizerSequence([
        NFC(),
        Replace("\n", NL_TOKEN),
        Replace("—", "-"),
        Replace(Regex(DIACRITICS_REGEX), ""),
        Replace(Regex(r'[“”«»]'), '"'),
        Replace(Regex(r'[’’‘]'), "'")
    ])

    tokenizer.pre_tokenizer = Whitespace()

    # Vocabulary Size
    vocab_size = num_merges + NUM_OF_CHAR + len(BYTE_FALLBACK_TOKENS)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=[UNK_TOKEN, NL_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN, MASK_TOKEN] + BYTE_FALLBACK_TOKENS,
        end_of_word_suffix=EW_TOKEN,
        # initial_alphabet=BYTE_FALLBACK_TOKENS,
    )

    tokenizer.train_from_iterator(
        corpus_iterator(corpus),
        trainer=trainer,
    )

    
    tokenizer.decoder = DecoderSequence([
        ByteFallback(),
        BPEDecoder(suffix=EW_TOKEN),
        decoders.Replace(NL_TOKEN, "\n"),
    ])

    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    print(f'Tokenizer saved to "{save_dir}/tokenizer.json"')


    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        sep_token=SEP_TOKEN,
        mask_token=MASK_TOKEN,
    )
    hf_tokenizer.save_pretrained(save_dir)
    print(f'Tokenizer files saved to "{save_dir}/"')

    return tokenizer


def encode(text: str, tokenizer) -> List[int]:
    return tokenizer.encode(text).ids


def decode(ids: List[int], tokenizer, show_markers: bool = False) -> str:
    decoded_text = tokenizer.decode(ids)

    if show_markers:
        return decoded_text

    return decoded_text.replace(PAD_TOKEN, "").replace(" ", "")


if __name__ == "__main__":
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
    tokenizer = AutoTokenizer.from_pretrained("tokenization/files")

    # Shout-out to " حسنين سمون ۽ بابار منگي "
    sample = "منهنجي دل کي رجهاءڻ لاء، رڳو مون سان ڪوڙ هياءي.\nوڏيون وڏيون ڳالهيون ڪياءي، وفا ته ڪانه ڪياءي." 
    print("Our sample text before tokenization")
    print("\n" + sample + "\n\n")

    print("Showing the tokens individually")

    ids = tokenizer.encode(sample)

    for i in range(len(ids)):
        print(tokenizer.decode(ids[i]))
    
    print("\nToken List")
    print(ids)
    
    print("\nReconstructed from Tokens")
    print(tokenizer.decode(ids))