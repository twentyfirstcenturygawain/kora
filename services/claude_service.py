import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class ClaudeService:
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = "claude-sonnet-4-6"

    def answer(self, question: str, context_chunks: list[tuple[str, float]]) -> str:
        if not context_chunks or context_chunks[0][1] < 0.05:
            return (
                "I couldn't find relevant information in the loaded documents "
                "to answer that. Try rephrasing, or switch to **chat mode** "
                "for general questions."
            )

        context = "\n\n---\n\n".join(
            f"[Relevance: {score:.2f}]\n{chunk}"
            for chunk, score in context_chunks
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            system="""You are a helpful customer support agent.
Answer the user's question using ONLY the provided context from the company's documentation.
Be concise, friendly, and accurate. Format your response clearly using markdown where helpful.
If the context doesn't fully answer the question, say so honestly.""",
            messages=[{
                "role": "user",
                "content": f"Context:\n\n{context}\n\nQuestion: {question}"
            }]
        )
        return response.content[0].text

    def free_chat(self, message: str) -> str:
        """General conversation mode — no document context."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            system="""You are Coda, a smart and friendly AI assistant. 
You can help with any question — analysis, writing, coding, strategy, or just conversation.
Be concise, warm, and genuinely helpful. Use markdown formatting where it adds clarity.""",
            messages=[{
                "role": "user",
                "content": message
            }]
        )
        return response.content[0].text