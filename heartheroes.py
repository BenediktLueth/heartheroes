import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Preprocessing

# import data
heart_disease = pd.read_csv("heart_2020_cleaned.csv")

# print all columns
print(heart_disease.describe())

# separate the target variable from the features
heart_disease_target = heart_disease['HeartDisease']
heart_disease_data = heart_disease.drop('HeartDisease', axis=1)

# plot the class distribution
class_distribution = pd.Series(heart_disease_target).value_counts()
plt.bar(class_distribution.index, class_distribution)
plt.xlabel("Heart disease")
plt.ylabel("Frequency")

# Preprocessing of numerical features

# list all numerical features
numerical_features = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]

# group data by target variable
heart_disease_grouped_by_class = heart_disease.groupby('HeartDisease')

# create a figure and specify its size
figure_histograms = plt.figure(figsize=(15, 5))
figure_histograms.text(0, 0.5, 'Frequency', va='center', rotation='vertical')

# go through all features and create one plot for each
figure_index = 1
for current_feature in numerical_features:
  axs = figure_histograms.add_subplot(1, 4, figure_index)
  for has_heart_disease, group in heart_disease_grouped_by_class:
    # plot the data points for the current group and feature combination
    axs.hist(group[current_feature], label=has_heart_disease)
  # set the axis labels of the current sub plot
  axs.set_xlabel(current_feature)
  # increase the figure index
  figure_index += 1

plt.tight_layout()
plt.show()
