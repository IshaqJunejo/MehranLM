import argparse
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def score_text(model, tokenizer, text: str, device: torch.device) -> tuple:
    ids = [tokenizer.bos_token_id] + tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids, labels=input_ids)
    n_pred = input_ids.size(1) - 1
    total_nll = outputs.loss.item() * n_pred
    ppl = math.exp(min(outputs.loss.item(), 20.0))
    bpc = total_nll / (max(1, len(text)) * math.log(2.0))
    return ppl, bpc


@torch.no_grad()
def generate_samples(model, tokenizer, prompt: str, temperatures, max_new_tokens: int, device: torch.device):
    ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    print(f"\nPrompt: {prompt}")
    print("=" * 70)
    for temp in temperatures:
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temp,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\n[Temperature = {temp}]\n{generated}")


def main():
    parser = argparse.ArgumentParser(description="Test trained MehranLM Transformer checkpoint.")
    parser.add_argument("--model_dir", type=str, default="checkpoints/mehranlm_transformer")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    prompts = [
        "سنڌ پاڪستان جو",
        "شاهه عبداللطيف ڀٽائي",
        "تاريخي طور تي",
    ]
    for prompt in prompts:
        generate_samples(model, tokenizer, prompt, [0.5, 0.8, 1.0], args.max_new_tokens, device)

    print("\n" + "=" * 70)
    sample_sentence = "سنڌو درياهه پاڪستان جو سڀ کان وڏو درياهه آهي."
    ppl, bpc = score_text(model, tokenizer, sample_sentence, device)
    print(f"Sample sentence: {sample_sentence}")
    print(f"Perplexity: {ppl:.2f} | BPC: {bpc:.4f}")


if __name__ == "__main__":
    main()
