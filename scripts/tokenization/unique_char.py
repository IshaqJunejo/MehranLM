import os

def main():
    text = ""
    input_dir = "../../Corpus/Cleaned/"
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_dir, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                text += f.read() + "\n"
    
    print(f"Total Characters: {len(text)}")

    chars = sorted(list(set(text)))

    print(f"Total Unique Character: {len(chars)}")

    # Write all unique characters to a file
    output_file = "./unique_chars.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(chars))

if __name__ == "__main__":
    main()