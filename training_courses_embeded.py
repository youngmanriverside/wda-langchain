from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from hf import model_name

# 確保 model_name 是一個 SentenceTransformer 對象
model = SentenceTransformer(model_name)


vectordb = Chroma(persist_directory="training_courses_embeded",embedding_function=model.encode)

# vectordb指定db後，使用as_retriever，在get_relevant會得到至少4個文件

retriever = vectordb.as_retriever()
docs = retriever.invoke('請問烘培課程有哪些?')


