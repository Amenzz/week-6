import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline

# Load FAISS index and metadata
index = faiss.read_index(r"C:\Users\Amenzz\Desktop\week-6\data\vector_store\complaint_chunks.index")
metadata_df = pd.read_csv(r"C:\Users\Amenzz\Desktop\week-6\data\vector_store\complaint_chunks_metadata.csv")

# Load models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text-generation", model="gpt2")

# Prompt template
PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}
Answer:
"""

# RAG core logic
def answer_question(question, top_k=5):
    question_embedding = embedding_model.encode([question])
    distances, indices = index.search(np.array(question_embedding).astype("float32"), top_k)
    retrieved_chunks = metadata_df.iloc[indices[0]]["text_chunk"].tolist()
    context = "\n\n".join(retrieved_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = generator(prompt[:1000], max_new_tokens=150, do_sample=False)[0]["generated_text"]
    return {
        "question": question,
        "generated_answer": response.split("Answer:")[-1].strip(),
        "retrieved_chunks": retrieved_chunks
    }

# Example usage
if __name__ == "__main__":
    result = answer_question("Why are people unhappy with Buy Now, Pay Later?")
    print("Question:", result["question"])
    print("\nAnswer:", result["generated_answer"])
    print("\nTop Retrieved Chunks:\n")
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        print(f"Chunk {i}:\n{chunk}\n{'-'*40}")
