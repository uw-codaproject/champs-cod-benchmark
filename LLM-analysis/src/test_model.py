from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

def main():
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    prompt = "You are a helpful assistant. Briefly say hello."
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
        )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()