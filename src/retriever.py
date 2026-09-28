import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Paths
# --------------------------------------------------

FAISS_PATH = "vectorstore/faiss.index"
CHUNKS_PATH = "vectorstore/chunks.pkl"


# --------------------------------------------------
# Load FAISS database
# --------------------------------------------------

index = faiss.read_index(FAISS_PATH)

print("FAISS index loaded successfully!")
print("Number of vectors:", index.ntotal)


# --------------------------------------------------
# Load text chunks
# --------------------------------------------------

with open(CHUNKS_PATH, "rb") as file:
    chunks = pickle.load(file)

print("Text chunks loaded successfully!")
print("Number of chunks:", len(chunks))


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")


# --------------------------------------------------
# Create retrieval function
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
# Test retrieval
# --------------------------------------------------

question = "What programming language is mainly used by NovaMind Analytics?"

results = retrieve_documents(question)

print("\nQuestion:")
print(question)

print("\nRetrieved Documents:")

for i, result in enumerate(results, start=1):

    print("\n------------------------------")
    print("Document", i)
    print("------------------------------")
    print(result)