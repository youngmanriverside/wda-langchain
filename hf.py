# import
from langchain_chroma import Chroma
import langchain_community.document_loaders as loaders
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

# load the document and split it into chunks
loader = loaders.TextLoader("test2.txt")
documents = loader.load()

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# load it into Chroma
db = Chroma.from_documents(docs, hf)

# save the database
# db = Chroma.from_documents(docs, hf, persist_directory="NovakDjokovic")

# query it
query = "How many masters title did Novak Djokovic have?"
docs = db.similarity_search(query)

# print results
print(docs[0].page_content)