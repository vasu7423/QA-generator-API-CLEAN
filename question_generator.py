from transformers import BertForQuestionAnswering, BertTokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch

# Load BERT model and tokenizer (for answer generation)
bert_model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForQuestionAnswering.from_pretrained(bert_model_name)

# Load T5 model and tokenizer (for question generation)
question_model_name = "valhalla/t5-small-qa-qg-hl"
tokenizer = T5Tokenizer.from_pretrained(question_model_name)
question_model = T5ForConditionalGeneration.from_pretrained(question_model_name)

# ✅ Function to generate multiple questions from input text using beam search
def generate_questions(text):
    input_text = f"generate questions: {text} </s>"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    outputs = question_model.generate(
        input_ids=input_ids,
        max_length=64,
        num_return_sequences=5,   # generate 5 questions
        num_beams=5,              # enable beam search
        early_stopping=True
    )

    questions = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
    return questions

# ✅ Function to generate answer using BERT
def generate_answer_bert(context, question):
    inputs = bert_tokenizer.encode_plus(
        question,
        context,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    input_ids = inputs["input_ids"].tolist()[0]

    with torch.no_grad():
        outputs = bert_model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

    # Find the best start and end token positions
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    # Ensure the answer is valid
    if start_index >= len(input_ids) or end_index >= len(input_ids) or start_index > end_index:
        return "Could not generate a meaningful answer."

    # Extract and decode the answer tokens
    answer_ids = input_ids[start_index : end_index + 1]
    answer = bert_tokenizer.decode(answer_ids, skip_special_tokens=True)

    return answer.strip()
