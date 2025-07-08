import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np
import uuid

# Load cleaned complaint data
df = pd.read_csv(r"C:\Users\Amenzz\Desktop\week-6\data\filtered_complaints.csv")
print(df.columns)
# Chunking function
def chunk_text(text, chunk_size=100, chunk_overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

# Apply chunking
chunked_data = []
for _, row in df.iterrows():
    chunks = chunk_text(row["cleaned_narrative"], 100, 20)
    for chunk in chunks:
        chunked_data.append({
            "complaint_id": row.get("Complaint ID", str(uuid.uuid4())),
            "product": row["Product"],
            "text_chunk": chunk
        })

chunked_df = pd.DataFrame(chunked_data)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(chunked_df["text_chunk"].tolist(), show_progress_bar=True)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

# Save the vector store
os.makedirs("vector_store", exist_ok=True)
faiss.write_index(index, "vector_store/complaint_chunks.index")
chunked_df.to_csv("vector_store/complaint_chunks_metadata.csv", index=False)

print("✅ Vector store and metadata saved in vector_store/")
