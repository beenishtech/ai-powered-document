import os
import chromadb
from sentence_transformers import SentenceTransformer

# 1. Documents folder se 5 files load karna
docs_dir = "documents"
documents = []
doc_names = []

if os.path.exists(docs_dir):
  for filename in os.listdir(docs_dir):
    if filename.endswith(".txt"):
      filepath = os.path.join(docs_dir, filename)
      with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
        documents.append(text)
        doc_names.append(filename)
else:
  print("Error: 'documents' folder not found!")

print(f"Total {len(documents)} documents successfully loaded.")

# 2. Documents ko chote text chunks mein divide karna
chunks = []
chunk_sources = []

for doc, name in zip(documents, doc_names):
  # Har document ko lines ya sentences mein split kar ke chunks banatay hain
  lines = doc.split("\n")
  for line in lines:
    if line.strip():  # Khali lines ignore karein
      chunks.append(line.strip())
      chunk_sources.append(name)

print(f"Kul {len(chunks)} text converted into chunks.")

# 3. Sentence-transformer model load karna embeddings ke liye
print("Sentence-transformer model loading...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. ChromaDB client initialize karna aur collection banana
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="ai_docs_collection")

# Chunks ke embeddings generate karna
print("Chunk's embeddings generating...")
embeddings = model.encode(chunks).tolist()

# Unique IDs create karna har chunk ke liye
ids = [f"chunk_{i}" for i in range(len(chunks))]

# ChromaDB mein data store karna
collection.add(
    documents=chunks,
    embeddings=embeddings,
    metadatas=[{"source": src} for src in chunk_sources],
    ids=ids,
)
print("Embeddings  are store in ChromaDB successfully\n")

# 5. User ka sawal define karna
user_question = "What is an intelligent agent in AI?"
print(f"--- USER QUESTION: {user_question} ---")

# 6. Sawal ko embedding mein convert kar ke top 3 similar chunks retrieve karna
question_embedding = model.encode([user_question]).tolist()

results = collection.query(query_embeddings=question_embedding, n_results=3)

# 7. Retrieved chunks aur unke distance/scores display karna
print("\nTop 3 Most Semantically Similar Chunks:")
print("=" * 50)

for i in range(len(results["documents"][0])):
  print(f"\n[Rank {i+1}]")
  print(f"Source File : {results['metadatas'][0][i]['source']}")
  print(f"Chunk Text  : {results['documents'][0][i]}")
  print(f"Distance    : {results['distances'][0][i]:.4f} (Lower is more similar)")