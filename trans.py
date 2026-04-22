from transformers import MarianMTModel, MarianTokenizer
import torch

MODEL_NAME = "Helsinki-NLP/opus-mt-bn-en"

def load_model():
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    return tokenizer, model

def translate_text(texts, tokenizer, model):
    translated = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model.generate(**inputs)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated.append(decoded)
    
    return translated

def main():
    # Load model
    tokenizer, model = load_model()

    # Read input file
    with open("input.txt", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Translate
    outputs = translate_text(lines, tokenizer, model)

    # Save output
    with open("output.txt", "w", encoding="utf-8") as f:
        for line in outputs:
            f.write(line + "\n")

    print("Translation completed. Saved to output.txt")

    # Print first sentence translation
    print("\nFirst translation:")
    print(outputs[0])

if __name__ == "__main__":
    main()
