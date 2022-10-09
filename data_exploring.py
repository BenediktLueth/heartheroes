import pandas as pd

heart_disease_data = pd.read_csv('heart_2020_cleaned.csv')

heart_disease_data.head()

heart_disease_data.shape

heart_disease_data.info()

heart_disease_data.describe()

heart_disease_data.describe(include=object)