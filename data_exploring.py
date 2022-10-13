import pandas as pd

heart_disease_data = pd.read_csv('heart_2020_cleaned.csv')

heart_disease_data.head()

heart_disease_data.shape

heart_disease_data.info()

heart_disease_data.describe()

heart_disease_data.describe(include=object)


# check for NaN values 

for col in heart_disease_data.columns:
    print("For column {} there are {} NaN values".format(col, heart_disease_data[col].isna().sum()))
    
# print distinct values for columns that are non-numerical 

for col in heart_disease_data.columns:
    if not heart_disease_data[col].dtype.kind in 'biufc':
        print("Unique values for column {}: {}".format(col, heart_disease_data[col].unique()))


binary_vals_dict = {1: 'Yes', 0: 'No'}
gen_health_dict = {1: 'Excellent', 2: 'Very good', 3: 'Good', 4: 'Fair', 5: 'Poor'}
sex_dict = {1: 'Male', 0: 'Female'}
diabetic_dict = {0: 'No', 1: 'Yes, during pregnancy', 2: 'No, borderline diabetes', 3: 'Yes'}
# using the average of age groups is a standard procedure in econmetric literature when dealing with age groups (assuming people are getting 110 years old max, but not many 
# so assuming on average 90 years in that age group)
age_category_dict = {'18-24': {'avg': 21, 'min': 18, 'max': 24}, '25-29': {'avg': 27, 'min': 25, 'max': 29},'30-34': {'avg': 32, 'min': 30, 'max': 34}, 
                     '35-39': {'avg': 37, 'min': 35, 'max': 39}, '40-44': {'avg': 42, 'min': 40, 'max': 44}, '45-49': {'avg': 47, 'min': 45, 'max': 49}, 
                     '50-54': {'avg': 52, 'min': 50, 'max': 54}, '55-59': {'avg': 57, 'min': 55, 'max': 59}, '60-64': {'avg': 62, 'min': 60, 'max': 64},
                     '65-69': {'avg': 67, 'min': 65, 'max': 69},  '70-74': {'avg': 72, 'min': 70, 'max': 74}, 
                     '75-79': {'avg': 77, 'min': 75, 'max': 79},  '80 or older': {'avg': 90, 'min': 80, 'max': 110},}
print("finished")