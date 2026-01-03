import os
import random
from tokenization import BPE_merges
from EDA import sentence_length

if __name__ == "__main__":
    merges = BPE_merges.load_merges("./tokenization/merges.txt")
    merge_ranks = BPE_merges.build_merge_ranks(merges)

    text = ""
    with open("../Corpus/Cleaned/sindhi_wiki_articles_cleaned.txt", "r", encoding="utf-8") as f:
        text += f.read()
    
    sentences = sentence_length.split_sentences(text)
    long_sentences = []
    
    print("Finished splitting the text into sentences")

    for sentence in sentences:
        tokens = sentence_length.tokenize_sentence(merge_ranks, sentence)

        if len(tokens) > 100:
            long_sentences.append((sentence, str(len(tokens))))
    
    print("Finished analyzing the length of sentences")

    sample_sentences = random.sample(long_sentences, k=25)

    with open("./QA/sentence_longer_than_75_tokens.txt", "w", encoding="utf-8") as f:
        for sentence in sample_sentences:
            f.write("[ " + sentence[1] + " ٽوڪن ]\n")
            f.write(sentence[0])
            f.write("\n------------------------\n")

    print("Finished writing to file")
