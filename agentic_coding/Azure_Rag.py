# Create a *RAG* for the Azure Preparation information in a .txt

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from langchain.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

import gradio as gr

# Load environment variables
load_dotenv(override=True)

# Load the source Azure text file
loader = TextLoader(file_path="azure_prep/Azure.txt", encoding="utf-8")
file = loader.load()

# Preview first part of the document
Number_of_characters = 2000
print(file[0].page_content[:Number_of_characters])

# Chunk the document for retrieval
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_documents(documents=file)

print("Chunks created:", len(chunks))

# (optional) inspect sample chunks
# print(chunks[:2])

# Constants for vector store
vector_db_name = "azure_prep"
all_mini = "all-MiniLM-L6-v2"
collection_name = "collection_azure_prep"

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name=all_mini)

# (optional) inspect embedding model
# print(embedding_model)

# Build (or rebuild) the Chroma vector DB
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=vector_db_name,
    collection_name=collection_name,
)

# Load the persisted vector DB
load_vector_db = Chroma(
    persist_directory=vector_db_name,
    embedding_function=embedding_model,
    collection_name=collection_name,
)

# Retriever (top‑k = 10)
retriever = load_vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Quick retrieval test
answer = retriever.invoke("Provide the Storage Services that Azure has")
content_answer = "".join([content.page_content for content in answer])

print("Retrieved context sample:")
print(content_answer[:500])

# Initialise Cerebras LLM
llm = ChatOpenAI(
    model="gpt-oss-120b",
    base_url=os.getenv("CEREBRAS_URL"),
    api_key=os.getenv("CEREBRAS_API_KEY"),
)

# (optional) Initialise Mistral LLM – not used in UI
llm_mistral = ChatOpenAI(
    model="devstral-2512",
    base_url=os.getenv("MISTRAL_URL"),
    api_key=os.getenv("MISTRAL_API_KEY"),
)

# Optional token count for retrieved context
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
tokens = encoding.encode(content_answer)
print("Tokens in retrieved context:", len(tokens))

# Demo message template
messages = [
    SystemMessage(
        "You are a professional Azure teacher, and you provide good explanations for the user's questions."
        " You provide good and understandable explanations to prepare a student for the Azure 900 fundamentals exam."
        " You base your answers from this information retrieved:"
    ),
    HumanMessage("hello world"),
]

# Gradio chatbot callback
def rag_chatbot(user_message: str, history: list):
    """
    Gradio callback handling a single turn of conversation.
    Streams the answer back to the UI.
    """
    history = [{"role": h["role"], "content": h["content"]} for h in history]

    # Retrieve relevant chunks
    retrieved_info = retriever.invoke(user_message)
    context = "".join([content.page_content for content in retrieved_info])

    # Build messages for LLM
    messages = [
        SystemMessage(
            "You are a professional Azure teacher, and you provide good explanations for the user's questions."
            " You provide good and understandable explanations to prepare a student for the Azure 900 fundamentals exam."
            " You base your answers from this information retrieved:\n{context}".format(context=context)
        ),
        HumanMessage(user_message),
    ] + history

    final_response = ""
    for chunk in llm.stream(messages):
        if chunk.content is not None:
            final_response += chunk.content
            yield final_response
        else:
            yield final_response

# Gradio interface definition
rag_chat = gr.ChatInterface(
    fn=rag_chatbot,
    chatbot=gr.Chatbot(height=1000, label="Azure 900 Exam"),
    title="Azure 900 Fundamentals Teacher",
    flagging_mode="never",
    examples=["Which are the Azure services?"],
    show_progress="minimal",
)

# Launch UI (guarded)
if __name__ == "__main__":
    rag_chat.launch(
        css="footer {display: none !important}",
        theme=gr.themes.Glass(),
    )

# Optional cleanup
# gr.close_all()   # uncomment if you need to kill existing servers
