from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from hf import model_name

#vectordb = chromadb.persistent_client.Chroma(persist_directory="/home/ren/user/project/wdaaichatbot/wda-langchain/training_courses_embeded/9390aec8-d846-4f38-81a1-03164aa6974d/index_metadata.pickle")
vectordb = Chroma(persist_directory="training_courses_embeded",embedding_function=model_name)

# vectordb指定db後，使用as_retriever，在get_relevant會得到至少4個文件

retriever = vectordb.as_retriever()
docs = retriever.invoke('請問烘培課程有哪些?')


