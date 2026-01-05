import os
from collections import Counter
import unicodedata

def main():
    input_dir = "../../Corpus/Cleaned/"
    char_freq = Counter()
    total_char = 0

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    char_freq.update(line)
                    total_char += len(line)
    
    print(f"Total Characters: {total_char}")
    print(f"Total Unique Characters: {len(char_freq)}")

    output_file = "./char_frequency.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Char\tFrequency\tUnicode Name\n")
        for ch, freq in sorted(char_freq.items(), key=lambda x: -x[1]):
            name = unicodedata.name(ch, "UNKNOWN")
            f.write(f"{ch}\t{freq}\t{name}\n")
    
    print(f"Character frequency report saved to {output_file}")

if __name__ == "__main__":
    main()
