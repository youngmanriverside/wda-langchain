from langchain_chroma import Chroma
import langchain_community.document_loaders as loaders
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# load the document and split it into chunks
loader = loaders.TextLoader("/home/ren/user/project/wdaaichatbot/wda-langchain/training_courses_10.csv")
documents = loader.load()

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0,length_function=len,is_separator_regex = False) 
docs = text_splitter.split_documents(documents)  #chunk_size=1800 ,training_courses.csv | chunk_size=500,traning_courses_10.csv 文本大小影響

# create the open-source embedding function
#model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_name = 'DMetaSoul/sbert-chinese-general-v2'
#model_name = 'DMetaSoul/sbert-chinese-qmc-finance-v1-distill'
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
#設定查詢回覆的數量

query = "烘培課程"
docs = db.similarity_search(query) #參數k是回覆的數量,預設4

# print results

print(docs[0].page_content)

