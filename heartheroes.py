# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import MinMaxScaler

# Preprocessing

# Import data
heart_disease = pd.read_csv("heart_2020_cleaned.csv")

# Preprocessing of numerical features

# Copy data to have the original one, if needed
heart_disease_numerical_preprocessed = heart_disease.copy()

# Print summary statistics of numerical features
print('Statistic of numerical features:')
print(heart_disease.describe())
print()

# List all numerical features
numerical_features = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]

# Normalization of numerical features

# Use MinMaxScaler to normalize numerical data, as half of the data isn't normally distributed
scaler = MinMaxScaler()
heart_disease_numerical_preprocessed[numerical_features] = \
    scaler.fit_transform(heart_disease_numerical_preprocessed[numerical_features])

print('Statistic of normalized numerical features:')
print(heart_disease_numerical_preprocessed.describe())

# Print correlation matrix of numerical features
print('\nCorrelation matrix of numerical features:')
print(heart_disease_numerical_preprocessed.corr())

# Plot distribution of numerical features, grouped by class
figure_distribution = plt.figure(figsize=(15, 8))
figure_distribution.canvas.set_window_title('Distribution of numerical features')
# Boxplot
figure_index = 1
for current_feature in numerical_features:
    ax = figure_distribution.add_subplot(2, 4, figure_index)
    heart_disease_numerical_preprocessed.boxplot(column=[current_feature], by='HeartDisease', ax=ax)
    figure_index += 1
# Histogram
for current_feature in numerical_features:
    ax = figure_distribution.add_subplot(2, 4, figure_index)
    for has_heart_disease, group in heart_disease_numerical_preprocessed.groupby('HeartDisease'):
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
combinations = itertools.combinations(numerical_features, 2)
# Go through all combinations and create one plot for each
for combination in combinations:
    # Add a subplot to the figure
    axs = figure_scatter.add_subplot(2, 3, figure_index)
    axs.scatter(heart_disease_numerical_preprocessed[combination[0]],
                heart_disease_numerical_preprocessed[combination[1]])
    # Set the axis labels of the current subplot
    axs.set_xlabel(combination[0])
    axs.set_ylabel(combination[1])
    # Increase the figure index (otherwise all plots are drawn in the first subplot)
    figure_index += 1

plt.tight_layout()
plt.show()
