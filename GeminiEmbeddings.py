import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from configparser import ConfigParser
import textwrap
import chromadb
import numpy as np
import pandas as pd
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain_chroma import Chroma


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


# Set up the DB
db = create_chroma_db("training_courses.csv", "geimini_training_courses_embeded")

# Set up the DB
#db = create_chroma_db(documents,"geimini_training_courses_embeded")

def get_relevant_passage(query, db):
  passage = db.query(query_texts=[query], n_results=1)['documents'][0][0]
  return passage


# Perform embedding search
passage = get_relevant_passage("提供烘培課程資訊", db)
print(passage)
