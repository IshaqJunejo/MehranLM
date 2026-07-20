import os
from dotenv import load_dotenv
from datasets import load_dataset

if __name__ == "__main__":
    load_dotenv()

    ds = load_dataset(
        "fahadqazi/Sindhi-Misspelled-Sentences",
        token=os.getenv("HF_TOKEN")
    )

    print(ds)

    NUM_OF_SENTENCES = 100000       # Loading only 100k rows of the data

    count = 0
    with open("../Corpus/Testing-data/sindhi_misspelled_sentences.txt", "w", encoding="utf-8") as f:
        for i in range(NUM_OF_SENTENCES):
            f.write(ds["train"]["incorrect"][i] + "\n")

            count += 1
            print(count, end="\r")
    
    print("Finished writing to \'sindhi_misspelled_sentences.txt\'")
    
    count = 0
    with open("../Corpus/Testing-data/sindhi_misspelling_corrected_sentences.txt", "w", encoding="utf-8") as f:
        for i in range(NUM_OF_SENTENCES):
            f.write(ds["train"]["correct"][i] + "\n")

            count += 1
            print(count, end="\r")
    
    print("Finished writing to \'sindhi_misspelling_corrected_sentences.txt\'")