import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import openai

# Load CSV (tab-separated)
df = pd.read_csv("jordan_transactions.csv", sep="\t", header=None)
df.columns = ["ID", "Mall", "Branch", "Date", "Quantity", "Price", "Type", "Status"]

# Create text corpus
rows = df.apply(lambda row: " | ".join(row.astype(str)), axis=1).tolist()

# Load model and generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(rows, convert_to_tensor=False)

# Create FAISS index
index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(embeddings)

# RAG query function
def query_csv(question, top_k=5):
    q_embed = model.encode([question])[0]
    D, I = index.search([q_embed], top_k)
    context = "\n".join([rows[i] for i in I[0]])
    
    prompt = f"""You are analyzing mall transactions. Use the following data to answer:
{context}

Question: {question}
Answer:"""
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Example usage
if __name__ == "__main__":
    question = "What is the total revenue from Y Mall?"
    answer = query_csv(question)
    print("Q:", question)
    print("A:", answer)
