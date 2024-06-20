import pandas as pd

# Load the data
data = pd.read_csv('wda-langchain/training_courses.csv')

# Export and save first 100 rows of csv data as a new csv file
data[:100].to_csv('training_courses_100.csv', index=False)

