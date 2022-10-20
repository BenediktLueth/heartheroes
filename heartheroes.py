# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import MinMaxScaler
from numpy import finfo

# Preprocessing

# Import data
heart_disease = pd.read_csv("heart_2020_cleaned.csv")

# Preprocessing of numerical features

# Print summary statistics of numerical features
print('Statistic of numerical features:')
print(heart_disease.describe())
print()

# List all numerical features
numerical_features = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]

# Normalization of numerical features

# Use MinMaxScaler, as half of the data isn't normally distributed
scaler = MinMaxScaler()

# copy data to have the original one, if needed
numerical_features_preprocessed = heart_disease.copy()

# use scaler to normalize numerical data
numerical_features_preprocessed[numerical_features] = \
  scaler.fit_transform(numerical_features_preprocessed[numerical_features])

print('Statistic of normalized numerical features:')
print(numerical_features_preprocessed.describe())

# Print correlation matrix of numerical features
print('\nCorrelation matrix of numerical features:')
print(numerical_features_preprocessed.corr(numeric_only=True))

# Plot distribution of numerical features, grouped by class
figure_distribution = plt.figure(figsize=(15, 8))
figure_distribution.canvas.set_window_title('Distribution of numerical features')
# Boxplot
figure_index = 1
for current_feature in numerical_features:
  ax = figure_distribution.add_subplot(2, 4, figure_index)
  numerical_features_preprocessed.boxplot(column=[current_feature], by='HeartDisease', ax=ax)
  figure_index += 1
# Histogram
for current_feature in numerical_features:
  ax = figure_distribution.add_subplot(2, 4, figure_index)
  for has_heart_disease, group in numerical_features_preprocessed.groupby('HeartDisease'):
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
#for current_x_feature in numerical_features:
#  for current_y_feature in numerical_features:
#    if current_x_feature != current_y_feature:
#      ax = figure_scatter.add_subplot(4, 3, figure_index)
#      ax.scatter(numerical_features_preprocessed[current_x_feature],
#                 numerical_features_preprocessed[current_y_feature], color='black')
#      plt.xlabel(current_x_feature)
#      plt.ylabel(current_y_feature)
#      figure_index += 1

# TODO @Benedikt, ich habe die Scatter Plots mit combinations abgelöst, weil wir die Verteilungen sonst doppelt haben.\
#  Schau mal, ob das für dich so passt.
combinations = itertools.combinations(numerical_features, 2)
# go through all combinations and create one plot for each
for combination in combinations:
  # add a sub plot to the figure
    axs = figure_scatter.add_subplot(2, 3, figure_index)
    axs.scatter(numerical_features_preprocessed[combination[0]], numerical_features_preprocessed[combination[1]])

    # set the axis labels of the current subplot
    axs.set_xlabel(combination[0])
    axs.set_ylabel(combination[1])

    # increase the figure index (otherwise all plots are drawn in the first subplot)
    figure_index += 1
plt.tight_layout()
plt.show()
