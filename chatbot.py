from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.llms.ollama import Ollama
from vectorstore import create_or_load_vector_store
from config import PROMPT_TEMPLATE

# Initialize memory for conversation history
memory = ConversationBufferMemory()

def chat_with_ollama(db,query_text: str, model):
    # Load or create vector store
    # db = create_or_load_vector_store(model_name)

    # Retrieve relevant documents from vector store
    results = db.similarity_search_with_score(query_text, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Retrieve past conversation from memory
    past_conversation = memory.load_memory_variables({})
    memory.save_context({"question": query_text}, {"response": context_text})

    # Combine past conversation with current context
    full_context = past_conversation.get('history', '') + "\n" + context_text

    # Prepare the prompt with memory and context
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=full_context, question=query_text)

    # Use Ollama's Mistral model for inference
    # model = Ollama(model=model_name)
    response_text = model.invoke(prompt)

    return response_text
