from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# --------------------------------------------------
# Load the generation model
# --------------------------------------------------

print("Loading generation model...")

MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Generation model loaded successfully!")


# --------------------------------------------------
# Generate answer from retrieved documents
# --------------------------------------------------

def generate_answer(question, documents):

    context = "\n\n".join(documents)

    prompt = f"""
Answer the question using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer


# --------------------------------------------------
# Test generation
# --------------------------------------------------

question = "What programming language is mainly used by NovaMind Analytics?"

documents = [
    """
    The main programming language used by the engineering teams is Python.
    SQL is used for querying and managing relational data.
    Power BI is used for interactive business dashboards.
    """
]

answer = generate_answer(
    question,
    documents
)

print("\nQuestion:")
print(question)

print("\nGenerated Answer:")
print(answer)