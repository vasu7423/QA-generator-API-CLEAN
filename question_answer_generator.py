from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re

# Load model and tokenizer
model_name = "google/flan-t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def generate_answer(question, context):
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **encoding,
        max_new_tokens=100,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove repetition and prompt prefix
    answer = answer.replace(prompt.strip(), "").strip()

    # Sanity check to avoid garbage like "gravity"
    if len(answer) < 5 or answer.lower() in ["gravity", "unknown", "(d)."]:
        return "Could not generate a meaningful answer."

    return answer
