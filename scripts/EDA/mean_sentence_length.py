import os
import re
from tokenization import BPE_merges

SENTENCE_SPLIT = re.compile(r'[.،؟!?.\n]+')

def split_sentences(text: str):
    sentences = SENTENCE_SPLIT.split(text)
    return [s.strip() for s in sentences if s.strip()]

def tokenize_sentence(merge_ranks: Dict[Tuple[str, str]], sentence: str):
    return BPE_merges.apply_bpe_to_text(merge_ranks, sentence)

def sentence_length_stat(filepath: str, merge_ranks: Dict[Tuple[str, str]]):
    text = ""
    if filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            text += f.read() + "\n"
    else:
        print(filepath + " is not a .txt file")
        return None
    
    sentences = split_sentences(text)

    if not sentences:
        print(filepath + " is an empty file!")
        return None
    
    distribution_label = ["< 5", "5 - 15", "15 - 25", "25 - 50", ">= 50"]
    distribution = [0, 0, 0, 0, 0]
    total_tokens = 0

    for sentence in sentences:
        tokens = tokenize_sentence(merge_ranks, sentence)

        if len(tokens) < 5:
            distribution[0] += 1
        elif 5 <= len(tokens) < 15:
            distribution[1] += 1
        elif 15 <= len(tokens) < 25:
            distribution[2] += 1
        elif 25 <= len(tokens) < 50:
            distribution[3] += 1
        elif len(tokens) >= 50:
            distribution[4] += 1
        
        total_tokens += len(tokens)
    
    print("Sentence Length Stats for " + filepath)
    print("Mean Sentence Length: " + str((total_tokens / len(sentences))))

    print("Number of sentences in each token-length category")
    for i in range(len(distribution)):
        print(distribution_label[i] + ": " + str(distribution[i]))
    
    print()

if __name__ == "__main__":
    merges = BPE_merges.load_merges("./tokenization/merges.txt")
    merge_ranks = BPE_merges.build_merge_ranks(merges)

    corpus_dir = "../Corpus/Cleaned/"
    for filename in os.listdir(corpus_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(corpus_dir, filename)
            sentence_length_stat(filepath, merge_ranks)