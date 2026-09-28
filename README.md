# 📚 RAG Document Assistant

A beginner-friendly **Retrieval-Augmented Generation (RAG)** application that allows users to ask questions about information stored in a PDF document.

The project combines **document retrieval** with **local AI text generation**. It uses **Sentence Transformers** to create embeddings, **FAISS** for similarity search, and **FLAN-T5** to generate answers based only on the retrieved document content.

---

## 🚀 Project Overview

Traditional question-answering systems may generate answers without having access to a user's specific documents.

This project solves that problem using **Retrieval-Augmented Generation (RAG)**.

The system:

1. Loads information from a PDF document.
2. Splits the document into smaller text chunks.
3. Converts the chunks into numerical embeddings.
4. Stores the embeddings in a FAISS vector index.
5. Retrieves the most relevant chunks for a user's question.
6. Passes the retrieved information to a local FLAN-T5 model.
7. Generates an answer based on the retrieved context.
8. Displays the answer through a Streamlit web application.

---

## 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** is an approach that combines information retrieval with text generation.

Instead of asking an AI model to answer a question using only its pretrained knowledge, RAG first retrieves relevant information from a knowledge source and then provides that information to the generation model.

### RAG Workflow

```text
PDF Document
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Sentence Transformer Embeddings
     ↓
FAISS Vector Store
     ↓
User Question
     ↓
Question Embedding
     ↓
Similarity Search
     ↓
Top Relevant Chunks
     ↓
FLAN-T5
     ↓
Generated Answer
```

---

## ✨ Features

* 📄 PDF document processing
* ✂️ Intelligent text chunking
* 🧠 Sentence Transformer embeddings
* 🔎 FAISS similarity search
* 🤖 Local FLAN-T5 text generation
* 📚 Context-based question answering
* 🖥️ Streamlit web interface
* 🔐 No paid API keys required
* 🌐 Can run locally
* 📦 Saved vector database for reuse
* 🔍 Displays retrieved documents for transparency

---

## 🛠️ Technologies Used

| Technology                | Purpose                   |
| ------------------------- | ------------------------- |
| Python                    | Main programming language |
| PyPDF                     | PDF text extraction       |
| LangChain Text Splitters  | Text chunking             |
| Sentence Transformers     | Text embeddings           |
| FAISS                     | Vector similarity search  |
| Hugging Face Transformers | Text generation           |
| FLAN-T5                   | Local language model      |
| Streamlit                 | Web application           |
| NumPy                     | Numerical operations      |
| Pickle                    | Saving text chunks        |

---

## 📁 Project Structure

```text
RAG_Document_Assistant/
│
├── data/
│   └── documents/
│       └── knowledge_document.pdf
│
├── src/
│   ├── retriever.py
│   └── generator.py
│
├── vectorstore/
│   ├── faiss.index
│   └── chunks.pkl
│
├── app.py
├── rag_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📄 Knowledge Document

The project uses a sample PDF knowledge base containing information about:

* NovaMind Analytics
* Core technologies
* AI and Data Science projects
* RAG systems
* Internship programs
* Frequently asked questions

The document is included in:

```text
data/documents/knowledge_document.pdf
```

---

## ⚙️ How the Project Works

### 1. Document Loading

The PDF is loaded using `pypdf`.

```python
from pypdf import PdfReader

reader = PdfReader(
    "data/documents/knowledge_document.pdf"
)
```

The text from all pages is extracted and combined.

---

### 2. Text Chunking

The extracted text is divided into smaller chunks using the LangChain recursive text splitter.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

The project generates **11 text chunks** from the knowledge document.

---

### 3. Creating Embeddings

Each text chunk is converted into a numerical vector using:

```text
all-MiniLM-L6-v2
```

The embedding model generates **384-dimensional vectors**.

These vectors allow the system to compare the semantic similarity between the user's question and the document content.

---

### 4. FAISS Vector Store

The embeddings are stored using FAISS.

```python
index = faiss.IndexFlatL2(
    embedding_dimension
)
```

FAISS performs similarity searches to identify the most relevant chunks for a question.

The saved vector store contains:

```text
vectorstore/faiss.index
vectorstore/chunks.pkl
```

---

### 5. Document Retrieval

When a user asks a question, the question is converted into an embedding.

FAISS then searches for the most similar document chunks.

The system retrieves the top **3 relevant chunks**.

---

### 6. Answer Generation

The retrieved chunks are combined into a context and passed to:

```text
google/flan-t5-small
```

The model generates an answer using the retrieved context.

The prompt instructs the model to answer only using the available document information.

---

## 🤖 Example

### Question

```text
What programming language is mainly used by NovaMind Analytics?
```

### Retrieved Information

The system retrieves a document chunk containing information about the company's core technologies.

### Answer

```text
Python
```

---

## 🖥️ Streamlit Application

The project includes a Streamlit interface where users can enter questions and receive answers.

Run the application using:

```bash
streamlit run app.py
```

The application provides:

* Question input
* Generated answer
* Retrieved document chunks

The retrieved chunks can be expanded using the **View Retrieved Documents** section.

---

## 💻 Installation

### 1. Clone the repository

```bash
git clone https://github.com/greeshma078/RAG_Document_Assistant.git
```

### 2. Move into the project directory

```bash
cd RAG_Document_Assistant
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

No paid API key is required.

---

## ▶️ Run the Project

### Test the complete RAG pipeline

```bash
python rag_pipeline.py
```

### Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📊 RAG Pipeline Components

| Component        | Model / Tool                   |
| ---------------- | ------------------------------ |
| PDF Loader       | PyPDF                          |
| Text Splitter    | RecursiveCharacterTextSplitter |
| Embedding Model  | all-MiniLM-L6-v2               |
| Vector Database  | FAISS                          |
| Retrieval        | FAISS Similarity Search        |
| Generation Model | google/flan-t5-small           |
| User Interface   | Streamlit                      |

---

## 🔍 Why FAISS?

FAISS is used to efficiently search through vector embeddings.

Instead of comparing a question with every piece of text manually, FAISS performs similarity search and returns the most relevant chunks.

This makes it suitable for building retrieval-based AI applications.

---

## 🔐 No Paid API Required

This project runs using locally downloaded open-source models.

It does **not require**:

* OpenAI API
* Gemini API
* Anthropic API
* Paid cloud AI services

The embedding model and FLAN-T5 model are downloaded from Hugging Face and run locally.

---

## 🎯 Learning Objectives

This project demonstrates practical understanding of:

* Natural Language Processing
* Text preprocessing
* Text chunking
* Semantic embeddings
* Vector databases
* Similarity search
* Retrieval-Augmented Generation
* Hugging Face Transformers
* Local LLM usage
* Streamlit application development
* Git and GitHub

---

## 🔮 Future Improvements

Possible future improvements include:

* Support for multiple PDF documents
* PDF upload through Streamlit
* Conversation history
* Improved retrieval techniques
* Source citations for every answer
* Hybrid keyword + semantic search
* Better generation models
* Reranking retrieved documents
* Chat-style interface
* Deployment to a cloud platform

---

## 👩‍💻 Author

**Greeshma Reddy**

B.Tech — Artificial Intelligence & Data Science

GitHub:
https://github.com/greeshma078

---

## ⭐ Project Highlights

This project demonstrates an end-to-end RAG workflow:

```text
Document
   ↓
Chunking
   ↓
Embeddings
   ↓
FAISS
   ↓
Retrieval
   ↓
FLAN-T5
   ↓
Answer
   ↓
Streamlit
```

Built as a practical learning project to understand how **Retrieval-Augmented Generation systems** work from document ingestion to final answer generation.
