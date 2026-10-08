import os
import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import streamlit as st

# Environment variables load karein
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Page Configuration
st.set_page_config(
    page_title="AI Document QA Assistant", page_icon="🤖", layout="wide"
)

st.title("📄 AI-Powered Document Question Answering System (RAG)")
st.write(
    "Ask questions about your technical documents using Semantic Search and"
    " Groq LLM."
)


# Vector Database load aur cache karne ke liye function
@st.cache_resource
def load_vector_db():
  docs_dir = "documents"
  documents = []
  doc_names = []

  if os.path.exists(docs_dir):
    for filename in os.listdir(docs_dir):
      if filename.endswith(".txt"):
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
          documents.append(f.read())
          doc_names.append(filename)

  chunks = []
  chunk_sources = []
  for doc, name in zip(documents, doc_names):
    for line in doc.split("\n"):
      if line.strip():
        chunks.append(line.strip())
        chunk_sources.append(name)

  # Model aur ChromaDB setup
  model = SentenceTransformer("all-MiniLM-L6-v2")
  chroma_client = chromadb.Client()
  collection = chroma_client.get_or_create_collection(name="streamlit_rag_docs")

  embeddings = model.encode(chunks).tolist()
  ids = [f"chunk_{i}" for i in range(len(chunks))]

  collection.add(
      documents=chunks,
      embeddings=embeddings,
      metadatas=[{"source": src} for src in chunk_sources],
      ids=ids,
  )
  return model, collection, doc_names


# Load database
model, collection, doc_names = load_vector_db()

# Sidebar mein loaded documents dikhayein
st.sidebar.header("📁 Loaded Documents")
for doc in doc_names:
  st.sidebar.write(f"- {doc}")

# Main interface - User Input
user_query = st.text_input(
    "Enter your question about the documents:",
    placeholder="e.g., What is an intelligent agent in AI?",
)

if user_query:
  with st.spinner("Searching documents and generating response..."):
    # 1. Semantic Search (Top 3 chunks retrieve karna)
    question_embedding = model.encode([user_query]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=3)

    retrieved_chunks = results["documents"][0]
    sources = results["metadatas"][0]
    distances = results["distances"][0]

    # 2. Context prepare karna RAG ke liye
    context = "\n".join(retrieved_chunks)

    # 3. Groq LLM ko context ke sath prompt bhejna
    rag_prompt = f"""
        You are an expert Technical Document Analyst. Answer the user's question accurately using ONLY the provided context below. If the answer is not in the context, state that you don't know.

        Context:
        {context}

        Question: {user_query}
        Answer:
        """

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "Provide professional, precise, and well-structured"
                    " answers based strictly on the provided context."
                ),
            },
            {"role": "user", "content": rag_prompt},
        ],
        temperature=0.3,
    )

    ai_answer = completion.choices[0].message.content

    # 4. UI par Results display karna
    st.subheader("💡 Generated Answer:")
    st.success(ai_answer)

    st.markdown("---")
    st.subheader("🔍 Retrieved Context Chunks (Semantic Search Results):")

    for i in range(len(retrieved_chunks)):
      st.info(
          f"**Rank {i+1} | Source: {sources[i]['source']} | Distance:"
          f" {distances[i]:.4f}**\n\n{retrieved_chunks[i]}"
      )