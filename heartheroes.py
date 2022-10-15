# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Preprocessing
# TODO BL: put all plots in one figure (?)

# Import data
heart_disease = pd.read_csv("heart_2020_cleaned.csv")

# Separate target variable from features
heart_disease_target = heart_disease['HeartDisease']
heart_disease_data = heart_disease.drop('HeartDisease', axis=1)

# Plot the class distribution
class_distribution = pd.Series(heart_disease_target).value_counts()
plt.bar(class_distribution.index, class_distribution)
plt.xlabel("Heart disease")
plt.ylabel("Frequency")

# Preprocessing of numerical features

# Print summary statistics for numerical features
print(heart_disease.describe())

# List all numerical features
numerical_features = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]

# Plot distribution for numerical features, grouped by class
figure = plt.figure(figsize=(15, 5))
# Boxplot
figure_index = 1
for current_feature in numerical_features:
    ax = figure.add_subplot(2, 4, figure_index)
    heart_disease.boxplot(column=[current_feature], by='HeartDisease', ax=ax)
    figure_index += 1
# Histogram
for current_feature in numerical_features:
    ax = figure.add_subplot(2, 4, figure_index)
    for has_heart_disease, group in heart_disease.groupby('HeartDisease'):
        # plot the data points for the current group and feature combination
        ax.hist(group[current_feature], label=has_heart_disease)
        ax.legend()
    figure_index += 1

# Format figures and display them
plt.tight_layout()
plt.suptitle('')
plt.show()
