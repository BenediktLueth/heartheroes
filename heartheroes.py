import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

# import data
heart_disease_data = pd.read_csv("heart_2020_cleaned.csv")

# print all columns
print(heart_disease_data.describe())

# plot histograms to get an overview of distribution
# TODO title, einmal amount of people und alle nebeneinander
# create a figure and specify its size
fig = plt.figure(figsize=(15, 10))

# go through all combinations and create one plot for each
numerical_feature = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]
figure_index = 1

heart_disease_data_grouped = heart_disease_data.groupby('HeartDisease')
for current_column in numerical_feature:
  # add a sub plot to the figure
  axs = fig.add_subplot(2, 3, figure_index)

  # group the data by HeartDisease (so we see the data points for Heart Diseases in different colours)
  for has_heart_disease, group in heart_disease_data_grouped:
    # plot the data points for the current group and feature combination
    axs.hist(group[current_column], label=has_heart_disease)

  # set the axis labels of the current sub plot
  axs.set_xlabel(current_column)
  axs.set_ylabel('Amount of People')

  # increase the figure index (otherwise all plots are drawn in the first subplot)
  figure_index += 1

# add a legend to the last sub plot
plt.tight_layout()
plt.legend()
plt.show()
