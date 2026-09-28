import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# --------------------------------------------------
# Paths
# --------------------------------------------------

FAISS_PATH = "vectorstore/faiss.index"
CHUNKS_PATH = "vectorstore/chunks.pkl"


# --------------------------------------------------
# Load FAISS index
# --------------------------------------------------

print("Loading FAISS index...")

index = faiss.read_index(FAISS_PATH)

print("FAISS index loaded successfully!")
print("Number of vectors:", index.ntotal)


# --------------------------------------------------
# Load text chunks
# --------------------------------------------------

print("\nLoading text chunks...")

with open(CHUNKS_PATH, "rb") as file:
    chunks = pickle.load(file)

print("Text chunks loaded successfully!")
print("Number of chunks:", len(chunks))


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")


# --------------------------------------------------
# Load generation model
# --------------------------------------------------

print("\nLoading generation model...")

MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

print("Generation model loaded successfully!")


# --------------------------------------------------
# Retrieval function
# --------------------------------------------------

def retrieve_documents(question, k=3):

    question_embedding = embedding_model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    distances, indices = index.search(
        question_embedding,
        k
    )

    retrieved_chunks = []

    for i in range(k):

        chunk_number = indices[0][i]

        retrieved_chunks.append(
            chunks[chunk_number]
        )

    return retrieved_chunks


# --------------------------------------------------
# Generation function
# --------------------------------------------------

def generate_answer(question, documents):

    context = "\n\n".join(documents)

    prompt = f"""
Answer the question using only the information provided in the context.

If the answer is not available in the context, say:
"I could not find this information in the document."

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
# Complete RAG pipeline
# --------------------------------------------------

def ask_question(question):

    documents = retrieve_documents(
        question,
        k=3
    )

    answer = generate_answer(
        question,
        documents
    )

    return answer, documents


# --------------------------------------------------
# Test the complete RAG pipeline
# --------------------------------------------------

question = "What programming language is mainly used by NovaMind Analytics?"

answer, documents = ask_question(question)

print("\n======================================")
print("RAG QUESTION")
print("======================================")

print(question)

print("\n======================================")
print("RETRIEVED DOCUMENTS")
print("======================================")

for i, document in enumerate(
    documents,
    start=1
):

    print(f"\nDocument {i}:")
    print(document)


print("\n======================================")
print("FINAL ANSWER")
print("======================================")

print(answer)