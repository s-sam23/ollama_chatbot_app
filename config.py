# Paths and constants for the chatbot project
VECTOR_DB_PATH = 'vector_db'
CHROMA_PATH = 'vector_db/chroma'
DATA_PATH = 'data'

# Chat prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

---

Answer the question based on the above context: {question}
"""
