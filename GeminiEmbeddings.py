import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from configparser import ConfigParser
import textwrap
import chromadb
import numpy as np
import pandas as pd
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_chroma import Chroma
import langchain_community.document_loaders as loaders
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


# Config Parser
config = ConfigParser()
config.read("config.ini")
genai.configure(api_key=config["Gemini"]["API_KEY"])

# Embed content
# result = genai.embed_content(
#     model="models/text-embedding-004",
#     content=[
#       'What is the meaning of life?',
#       'How much wood would a woodchuck chuck?',
#       'How does the brain work?'])

class GeminiEmbeddingFunction(EmbeddingFunction):
  def __call__(self, input: Documents) -> Embeddings:
    model = 'models/text-embedding-004'
    title = "Custom query"
    return genai.embed_content(model=model,
                                content=input,
                                task_type="retrieval_document",
                                title=title)["embedding"]


def create_chroma_db(documents, name):
  chroma_client = chromadb.Client()
  db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())

  for i, d in enumerate(documents):
    db.add(
      documents=d,
      ids=str(i)
    )
  return db

# load the document and split it into chunks
loader = loaders.TextLoader("/home/ren/user/project/wdaaichatbot/wda-langchain/training_courses.csv")
documents = loader.load()

 # split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0,length_function=len,is_separator_regex = False) 
docs = text_splitter.split_documents(documents)  #chunk_size=1800 ,training_courses.csv | chunk_size=500,traning_courses_10.csv 文本大小影響

# Set up the DB
db = create_chroma_db(docs,"geimini_training_courses_embeded")

def get_relevant_passage(query, db):
  passage = db.query(query_texts=[query], n_results=1)['documents'][0][0]
  return passage


# Perform embedding search
passage = get_relevant_passage("提供水電課程資訊", db)
print(passage)
