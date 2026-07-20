import os
from dotenv import load_dotenv
from transformers import AutoTokenizer
from tokenizers import Tokenizer

load_dotenv()

REPO_ID = "meta-llama/Llama-3.1-8B" 
LOCAL_OUTPUT_DIR = "./tokenizer_benchmarking/llama_tokenizer_files"

def download_private_tokenizer():
    print(f"Attempting to fetch tokenizer from gated repo: {REPO_ID}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            REPO_ID, 
            token=os.getenv("HF_TOKEN") 
        )
        
        tokenizer.save_pretrained(LOCAL_OUTPUT_DIR)
        
        print(f"Tokenizer files downloaded and saved to: {LOCAL_OUTPUT_DIR}")
        # print("Files created typically include: tokenizer.json, tokenizer_config.json, etc.")
        
    except Exception as e:
        print(f"Failed to download tokenizer. Error: {e}")
        # print("Double-check that your HF_TOKEN is valid and has 'Read' access to the repository.")

if __name__ == "__main__":
    download_private_tokenizer()
