import pandas as pd
import pathlib
import matplotlib.pyplot as plt
from torch import threshold

folder_path = pathlib.Path("data")

#! Use the pandas and pathlib.Path to load all 5 .csv files into an array as DataFrames (that is, 5 separate dataframes stored in a list)
dataframes = [pd.read_csv(file) for file in folder_path.glob("*.csv")]

# split the first dataframe in half and use the first half for validation and the second half for testing
# use the other 4 dataframes for training
for i, df in enumerate(dataframes):
    #drop the first and last column and first row 
    dataframes[i] = df.iloc[0:, 1:-1]

df_valid = dataframes[0].iloc[:len(dataframes[0])//2]
df_test = dataframes[0].iloc[len(dataframes[0])//2:]
df_train = pd.concat(dataframes[1:])


# for df df_valid and df_train, the threshold is 2000 < x < 100000. do nothing for df_test
def clean_series(series, lower_threshold, upper_threshold):
    mean_value = series.mean()
    return series.where((series.abs() >= lower_threshold) & (series.abs() <= upper_threshold), mean_value)

df_valid = df_valid.apply(clean_series, lower_threshold=2000, upper_threshold=100000)
df_train = df_train.apply(clean_series, lower_threshold=2000, upper_threshold=100000)

plt.plot(df_valid)
plt.show()
plt.plot(df_test)
plt.show()
plt.plot(df_train)
plt.show()
