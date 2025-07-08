from rag_pipeline import answer_question
import pandas as pd

sample_questions = [
    "Why are people unhappy with Buy Now, Pay Later?",
    "What complaints are common about Credit Cards?",
    "Are there issues with Savings Accounts?",
    "What do users say about Personal Loans?",
    "Is there any fraud reported with Money Transfers?",
    "Why are people frustrated with late payments?",
    "What concerns do customers have about interest rates?",
    "Are there technical problems mentioned in complaints?",
    "Do people report unauthorized charges?",
    "What are users saying about the mobile app?"
]

# Run evaluation
results = []

for question in sample_questions:
    output = answer_question(question)
    results.append({
        "Question": output["question"],
        "Generated Answer": output["generated_answer"],
        "Retrieved Source 1": output["retrieved_chunks"][0] if output["retrieved_chunks"] else "",
        "Retrieved Source 2": output["retrieved_chunks"][1] if len(output["retrieved_chunks"]) > 1 else ""
    })

# Save as CSV (for manual scoring)
df = pd.DataFrame(results)
df.to_csv("data/evaluation_results.csv", index=False)

print("✅ Evaluation results saved to data/evaluation_results.csv")
