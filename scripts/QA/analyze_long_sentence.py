import os
import random
from tokenization import tokenizer
from EDA import sentence_length

if __name__ == "__main__":
    token_dict = tokenizer.load_tokens("./tokenization/tokenizer.json")

    text = ""
    with open("../Corpus/Cleaned/sindhi_wiki_articles_cleaned.txt", "r", encoding="utf-8") as f:
        text += f.read()
    
    sentences = sentence_length.split_sentences(text)
    long_sentences = []
    
    print("Finished splitting the text into sentences")

    for sentence in sentences:
        tokens = sentence_length.tokenize_sentence(sentence, token_dict)

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
