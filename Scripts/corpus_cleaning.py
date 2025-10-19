import os
import re


def clean_text(text: str) -> str:
    # Remove MediaWiki or HTML markup
    text = re.sub(r'\[\[.*?\]\]', '', text)       # [[links]]
    text = re.sub(r'\{\{.*?\}\}', '', text)       # {{templates}}
    text = re.sub(r'<.*?>', '', text)             # <tags>
    text = re.sub(r'={2,}', '', text)             # == section titles ==
    text = re.sub(r'\|.*?\n', '\n', text)         # tables / pipes

    # Remove URLs, email addresses, and numbers
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)

    # Remove stray punctuation or excessive symbols
    text = re.sub(r'[“”"\'–—_•·<>•=*#|]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Trim whitespace
    text = text.strip()

    return text


def is_sindhi_text(s: str) -> bool:
    # Return True if the line is mostly Sindhi text
    sindhi_chars = sum(1 for c in s if '\u0600' <= c <= '\u08FF')
    latin_chars = sum(1 for c in s if 'a' <= c.lower() <= 'z')
    
    return sindhi_chars > max(latin_chars * 2, 5)


def clean_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        cleaned = clean_text(line)
        if cleaned and is_sindhi_text(cleaned):
            cleaned_lines.append(cleaned)

    # Write cleaned text to output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_lines))

    print(f"Cleaned {len(cleaned_lines)} lines to {output_path}")


def main():
    input_dir = "../Corpus/Raw/"
    output_dir = "../Corpus/Cleaned/"

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            in_path = os.path.join(input_dir, filename)
            out_path = os.path.join(output_dir, filename.replace(".txt", "_cleaned.txt"))
            clean_file(in_path, out_path)


if __name__ == "__main__":
    main()
