#轉換檔案格式爲tsv
#讀取training_courses.csv'

import csv
import pandas as pd

# 讀取csv檔案
df = pd.read_csv('training_courses.csv')

# 將csv檔案轉換爲tsv檔案

df.to_csv('training_courses.tsv', sep='\t', index=False)

