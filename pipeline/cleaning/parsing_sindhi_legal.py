import os
import pandas as pd

df = pd.read_csv("../Corpus/Raw/Sindhi_Legal_Dataset.csv")

count = 0
for _, row in df.iterrows():
    question = str(row["input_sd"])
    answer = str(row["output_sd"])

    with open(f"../Corpus/Raw/sindhi_legal_dataset.txt", "a", encoding="utf-8") as f:
        f.write(question + "\n\n")
        f.write(answer + "\n\n")
    
    count += 1
    print(count, end="\r")

print("Finished parsing the Legal Data")
