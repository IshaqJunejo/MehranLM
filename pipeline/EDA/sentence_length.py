import os
import re
from tokenization import tokenizer

SENTENCE_SPLIT = re.compile(r'[-.،۔؟!?.؛;\n]+')

def split_sentences(text: str):
    sentences = SENTENCE_SPLIT.split(text)
    return [s.strip() for s in sentences if s.strip()]

def tokenize_sentence(sentence: str, token_dict):
    return tokenizer.encode(sentence, token_dict)

def sentence_length_stat(filepath: str, token_dict):
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
    

    token_len_distribution = []
    total_tokens = 0

    for sentence in sentences:
        tokens = tokenize_sentence(sentence, token_dict)

        token_len_distribution.append(len(tokens))        
        total_tokens += len(tokens)
    
    token_len_distribution = sorted(token_len_distribution)
    
    print("Sentence Length Stats for " + filepath)
    print("Mean Sentence Length: " + str((total_tokens / len(sentences))))
    print("Total Tokens in the file: " + str(total_tokens))

    print("\nToken Length In Percentiles")
    print("50%: " + str(token_len_distribution[int(len(token_len_distribution) * 0.5)]))
    print("75%: " + str(token_len_distribution[int(len(token_len_distribution) * 0.75)]))
    print("90%: " + str(token_len_distribution[int(len(token_len_distribution) * 0.90)]))
    print("95%: " + str(token_len_distribution[int(len(token_len_distribution) * 0.95)]))
    print("99%: " + str(token_len_distribution[int(len(token_len_distribution) * 0.99)]))
    
    print("---")

if __name__ == "__main__":
    token_dict = tokenizer.load_tokens("./tokenization/tokenizer.json")

    corpus_dir = "../Corpus/Cleaned/"
    for filename in os.listdir(corpus_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(corpus_dir, filename)
            sentence_length_stat(filepath, token_dict)
