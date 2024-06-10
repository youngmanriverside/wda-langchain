from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, AzureOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import bs4
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


model = "gpt-4o"
api_version = "2024-05-13"
azure_endpoint = "https://openai-wenshin.openai.azure.com/"
api_key = "bfe36c061de0450daa111f54cf1b11ad"
llm = AzureChatOpenAI(model=model,
                      api_version=api_version,
                      azure_endpoint=azure_endpoint,
                      api_key=api_key)

loader = WebBaseLoader(
    web_path="https://lilianweng.github.io/posts/2023-06-23-agent/",
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
splits = text_splitter.split_documents(documents)

# Embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Creating Chroma database")
vectorstore = Chroma.from_documents(documents=splits, embedding=hf)

retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")