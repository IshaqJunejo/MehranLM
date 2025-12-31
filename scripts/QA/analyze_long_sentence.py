import os
import random
from tokenization import BPE_merges
from EDA import mean_sentence_length

if __name__ == "__main__":
    merges = BPE_merges.load_merges("./tokenization/merges.txt")
    merge_ranks = BPE_merges.build_merge_ranks(merges)

    text = ""
    with open("../Corpus/Cleaned/sindhi_wiki_articles_cleaned.txt", "r", encoding="utf-8") as f:
        text += f.read()
    
    sentences = mean_sentence_length.split_sentences(text)
    long_sentences = []
    
    print("Finished splitting the text into sentences")

    for sentence in sentences:
        tokens = mean_sentence_length.tokenize_sentence(merge_ranks, sentence)

        if len(tokens) > 75:
            long_sentences.append(sentence)
    
    print("Finished analyzing the length of sentences")

    sample_sentences = random.sample(long_sentences, k=25)

    with open("./QA/sentence_longer_than_75_tokens.txt", "w", encoding="utf-8") as f:
        for sentence in sample_sentences:
            f.write(sentence + "\n------------------\n")

    print("Finished writing to file")
