# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from numpy import finfo

# Preprocessing

# Import data
heart_disease = pd.read_csv("heart_2020_cleaned.csv")

# Preprocessing of numerical features

# Print summary statistics of numerical features
print('Statistic of numerical features:')
print(heart_disease.describe())

# Print correlation matrix of numerical features
print('\nCorrelation matrix of numerical features:')
print(heart_disease.corr())

# List all numerical features
numerical_features = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]

# Plot distribution of numerical features, grouped by class
figure_distribution = plt.figure(figsize=(15, 8))
figure_distribution.canvas.set_window_title('Distribution of numerical features')
# Boxplot
figure_index = 1
for current_feature in numerical_features:
    ax = figure_distribution.add_subplot(2, 4, figure_index)
    heart_disease.boxplot(column=[current_feature], by='HeartDisease', ax=ax)
    figure_index += 1
# Histogram
for current_feature in numerical_features:
    ax = figure_distribution.add_subplot(2, 4, figure_index)
    for has_heart_disease, group in heart_disease.groupby('HeartDisease'):
        # plot the data points for the current group and feature combination
        ax.hist(group[current_feature], label=has_heart_disease)
        ax.legend()
    figure_index += 1
plt.suptitle('')
plt.tight_layout()

# Plot scatters of numerical features
figure_scatter = plt.figure(figsize=(15, 8))
figure_scatter.canvas.set_window_title('Scatter plot of numerical features')
figure_index = 1
for current_x_feature in numerical_features:
    for current_y_feature in numerical_features:
        if current_x_feature != current_y_feature:
            ax = figure_scatter.add_subplot(4, 3, figure_index)
            ax.scatter(heart_disease[current_x_feature], heart_disease[current_y_feature], color='black')
            plt.xlabel(current_x_feature)
            plt.ylabel(current_y_feature)
            figure_index += 1
plt.tight_layout()

plt.show()

# TODO @FD implement normalization of numerical features
# Normalization of numerical features

