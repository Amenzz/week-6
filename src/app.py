import gradio as gr
from rag_pipeline import answer_question

def ask_rag(question):
    if not question.strip():
        return "❗ Please enter a question.", ""

    result = answer_question(question)
    answer = result["generated_answer"]
    sources = "\n\n".join([f"• {chunk}" for chunk in result["retrieved_chunks"][:2]])

    return answer, sources

# Create Gradio interface
iface = gr.Interface(
    fn=ask_rag,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about customer complaints..."),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Retrieved Source Chunks")
    ],
    title="CrediTrust Complaint Analyst",
    description="Ask any question about customer complaints. Get AI-generated answers with context."
)

if __name__ == "__main__":
    iface.launch()
