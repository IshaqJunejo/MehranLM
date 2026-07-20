from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
import time
import json
from pathlib import Path
import re
import os
from dotenv import load_dotenv

load_dotenv()

def benchmark_tokenizer(tokenizer, model_name, text):
    vocab_size = len(tokenizer)

    # Warm-up
    tokenizer.encode(text)

    start = time.perf_counter()
    tokens = tokenizer.encode(text)
    end = time.perf_counter()
    time_elapsed = end - start
    
    token_count = len(tokens)
    fertility_rate = token_count / len(text.split())
    compression_ratio = len(text.encode('utf-8')) / token_count
    throughput = len(text) / time_elapsed

    words = text.split()
    total_words = len(words)

    # OOV Rate and STRR
    unk_token = tokenizer.unk_token_id
    oov_count = 0
    single_token_count = 0

    for word in words:
        token_ids = tokenizer.encode(word, add_special_tokens=False)
        if unk_token in token_ids:
            oov_count += 1
        if len(token_ids) == 1:
            single_token_count += 1
    
    oov_rate = (oov_count / total_words) if total_words > 0 else 0
    strr = (single_token_count / total_words) if total_words > 0 else 0
    
    # Byte Fallback Rate
    byte_fallback_count = 0

    byte_pattern = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
    byte_ids = {
        tok_id for tok, tok_id in tokenizer.get_vocab().items() 
        if byte_pattern.match(tok)
    }

    byte_fallback_count += sum(1 for tid in tokens if tid in byte_ids)
    
    byte_fallback_rate = (byte_fallback_count / token_count) if token_count > 0 else 0
    
    return {
        "model": model_name,
        "vocab_size": vocab_size,
        "token_count": token_count,
        "fertility_rate": round(fertility_rate, 4),
        "time_elapsed": round(time_elapsed, 4),
        "throughput_chars_sec": round(throughput, 4),
        "chars_per_token": round(len(text) / token_count, 2) if token_count else 0,
        "bytes_per_token": round(compression_ratio, 4),
        "unk_count": oov_count,
        "out_of_vocab_rate": round(oov_rate, 4),
        "single_token_word_count": single_token_count,
        "single_token_retention_rate": round(strr, 4),
        "byte_fallback_count": byte_fallback_count,
        "byte_fallback_rate": round(byte_fallback_rate, 4)
    }


if __name__ == "__main__":
    # Loading testing data
    text_file_01 = Path("../Corpus/Testing-data/sindhi_sentiment_parsed.txt")
    sentiment_data = text_file_01.read_text(encoding="utf-8")

    text_file_02 = Path("../Corpus/Testing-data/sindhi_misspelled_sentences.txt")
    misspelling_data = text_file_02.read_text(encoding="utf-8")

    text_file_03 = Path("../Corpus/Testing-data/sindhi_misspelling_corrected_sentences.txt")
    misspelling_corrected_data = text_file_03.read_text(encoding="utf-8")

    # List of models
    models = [
        # "IshaqueJunejo/MehranLM",             # To be Accessed locally
        "aakashMeghwar01/SindhiLM-Tokenizer-v1",
        "aakashMeghwar01/SindhiLM-Tokenizer-v2",
        "aakashMeghwar01/SindhiLM-Tokenizer-v3",
        "fahadqazi/Sindhi-BPE-Tokenizer",

        "Qwen/Qwen2.5-7B",
        # "meta-llama/Meta-Llama-3.1-8B",       # Gated Repo (Downloaded the Tokenizer) - To be Accessed locally
        "google-bert/bert-base-multilingual-uncased",
        "FacebookAI/xlm-roberta-base",
    ]

    names = [
        "", "", "", "", "", "", "",
        "MehranLM-tokenizer",
        "meta-llama/Meta-Llama-3.1-8B",
        "MehranLM-tokenizer-with-diacritics"
    ]

    print("Loading tokenizers...")

    tokenizer_models = []

    for model in models:
        current_tokenizer = AutoTokenizer.from_pretrained(
            model,
            token=os.getenv("HF_TOKEN")
        )
        tokenizer_models.append(current_tokenizer)
    
    mehranlm_tokenizer = AutoTokenizer.from_pretrained("tokenization/files")
    tokenizer_models.append(mehranlm_tokenizer)

    llama_tokenizer = AutoTokenizer.from_pretrained("tokenizer_benchmarking/llama_tokenizer_files")
    tokenizer_models.append(llama_tokenizer)

    mehranlm_tokenizer_2 = AutoTokenizer.from_pretrained("tokenizer_benchmarking/mehranLM_tokenizer_files_with_diacritics")
    tokenizer_models.append(mehranlm_tokenizer_2)

    print("Tokenizers loaded!")

    # Test data and the name of the files
    test_data = [
        ["sentiment-data", sentiment_data],
        ["misspelling-data", misspelling_data],
        ["misspelling-corrected-data", misspelling_corrected_data]
    ]

    for j in range(len(test_data)):
        print(f"\nTesting on {test_data[j][0]}...")

        # Testing the models
        results = []

        for i in range(len(tokenizer_models)):
            if i < len(models):
                try:
                    print(f"Benchmarking {models[i]}...")
                    res = benchmark_tokenizer(tokenizer_models[i], models[i], test_data[j][1])
                    results.append(res)
                except Exception as e:
                    print(f"Error with {models[i]}: {e}")
            else:
                try:
                    print(f"Benchmarking {names[i]}...")
                    res = benchmark_tokenizer(tokenizer_models[i], names[i], test_data[j][1])
                    results.append(res)
                except Exception as e:
                    print(f"Error with {names[i]}: {e}")

        # Sort by token efficiency (fewer tokens better)
        results.sort(key=lambda x: x["token_count"])

        # Print and save results
        # print(json.dumps(results, indent=2))
        with open(f"tokenizer_benchmarking/results/results-on-{test_data[j][0]}.json", "w") as f:
            json.dump(results, f, indent=2)
