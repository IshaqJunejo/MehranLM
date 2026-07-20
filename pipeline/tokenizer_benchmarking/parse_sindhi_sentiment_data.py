import os
import pandas as pd

df = pd.read_csv("../Corpus/Testing-data/sindhi_sentiment_cleaned.csv")

with open("../Corpus/Testing-data/sindhi_sentiment_parsed.txt", "w", encoding="utf-8") as f:
    f.write("")

count = 0
for _, row in df.iterrows():
    statement = str(row["Sindhi Text"])

    with open("../Corpus/Testing-data/sindhi_sentiment_parsed.txt", "a", encoding="utf-8") as f:
        f.write(statement + "\n")
    
    count += 1
    print(count, end="\r")

print(f"Finished parsing the \'Sindhi Sentiment Dataset\' with {count} statements")