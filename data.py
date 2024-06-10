import pandas as pd

# Load the csv file
df = pd.read_csv("training_courses.csv")

# Save first 10 rows as a new csv file
df.head(10).to_csv("training_courses_10.csv", index=False)
