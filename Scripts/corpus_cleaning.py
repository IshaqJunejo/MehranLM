import os
import re
import html

# Zero width characters
ZERO_WIDTH = ['\u200B', '\u200C', '\u200D', '\uFEFF', '\u200E', '\u200F', '\u202A', '\u202B', '\u202C', '\u202D', '\u202E']

# Arabic character ranges (Unicode)
ARABIC_RANGES = [
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
]

# Is the given character in Unicode ranges of Arabic characters
def is_arabic_char(ch: int) -> bool:
    for a, b in ARABIC_RANGES:
        if a <= ch and ch <= b:
            return True
    
    return False

# Remove Zero Width characters
def remove_zero_width(s: str) -> str:
    for ch in ZERO_WIDTH:
        s = s.replace(ch, '')

    return s

def clean_text(text: str) -> str:
    text = html.unescape(text)

    # Remove weird leftover symbols like " &;/&;"
    text = re.sub(r'[&;$]+', ' ', text)

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

    # Remove Latin characters
    text = re.sub(r'[A-Za-z]', '', text)

    # Remove Zero width characters
    text = remove_zero_width(text)

    # Trim whitespace
    text = text.strip()

    return text


def is_sindhi_text(s: str) -> bool:
    stripped = s.strip()

    if not stripped:
        return False
    
    sindhi_chars = sum(1 for c in s if is_arabic_char(ord(c)))
    total_chars = sum(1 for c in stripped if not c.isspace())
    
    if total_chars == 0:
        return False

    # Return True if more than 40% of characters in the line are Sindhi characters
    return (sindhi_chars / total_chars) > 0.4


def clean_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Cleaning each line individually
    cleaned_lines = []
    for line in lines:
        cleaned = clean_text(line)
        if cleaned and is_sindhi_text(cleaned):
            cleaned_lines.append(cleaned)
    
    # De-duplication
    seen = set()
    unique_lines = []
    for line in cleaned_lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    # Write cleaned text to output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(unique_lines))

    print(f"Cleaned {len(unique_lines)} lines to {output_path}")


def main():
    input_dir = "../Corpus/Raw/Large/"          # This Directory is mentioned in .gitignore
    # input_dir = "../Corpus/Raw/"
    output_dir = "../Corpus/Cleaned/"

    # input_dir = "../Corpus/Private/Raw/"        # This Directory is mentioned in .gitignore
    # output_dir = "../Corpus/Private/Cleaned/"   # This Directory is mentioned in .gitignore

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            in_path = os.path.join(input_dir, filename)
            out_path = os.path.join(output_dir, filename.replace(".txt", "_cleaned.txt"))
            clean_file(in_path, out_path)


if __name__ == "__main__":
    main()
