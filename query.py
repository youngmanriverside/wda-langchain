from langchain_chroma import Chroma

# embedding function
from hf import hf

# load the database
db = Chroma(persist_directory="training_courses_10", embedding_function=hf)

# query it
query = "烘焙課程"
docs = db.similarity_search(query)

# print results
print(docs[0].page_content)
