import pandas as pd
import matplotlib.pyplot as plt

"""
Pandas is used to analyze data. It is a powerful data manipulation library (analyzing, cleaning, exploring, and manipulating data).
"""


def pandas_usage():
    pandas_series()
    pandas_dataframe()
    pandas_analyze()
    pandas_plot()
    pass


def pandas_plot():
    dataframe_food = pd.read_json("data/food_data.json")

    dataframe_food.plot(x="Duration", y="Pulse", kind="bar")  # plot bar chart
    plt.show()  # show plot

    dataframe_food.plot(x="Duration", y="Pulse", kind="scatter")  # plot scatter
    plt.show()  # show plot

    dataframe_food["Duration"].plot(x="Duration", kind="hist")  # plot histogram
    plt.show()  # show plot
    pass


def pandas_analyze():
    dataframe_food = pd.read_json("data/food_data.json")
    print(f"dataframe_food head: \n{dataframe_food.head()}")  # first 5 rows
    print(f"dataframe_food tail: \n{dataframe_food.tail()}")  # last 5 rows
    print(f"dataframe_food info: \n{dataframe_food.info()}")  # last 5 rows

    new_dataframe_food_remove_rows_with_null = dataframe_food.dropna()  # remove rows with null values
    dataframe_food.fillna("NA", inplace=True)  # fill null values with "NA"
    dataframe_food["Calories"].fillna("NA", inplace=True)  # fill null values with "NA" only for "Calories" column
    mean = dataframe_food["Calories"].mean()  # mean of "Calories" column
    mode = dataframe_food["Calories"].mode()  # mode of "Calories" column
    median = dataframe_food["Calories"].median()  # median of "Calories" column
    bool_duplicated_series = dataframe_food.duplicated()  # check for duplicates, returns boolean series with True for duplicates
    dataframe_food.drop_duplicates(inplace=True)  # remove duplicates

    dataframe_correlation = dataframe_food.corr()  # correlation between columns, 0-1: 0 means no correlation, 1 means perfect correlation
    print(f"dataframe_correlation: \n{dataframe_correlation}")
    pass


def pandas_series():  # 1D array representing a column or row
    a = [-1, 7, 2, 4]  # 1st value treated as label
    # num_series = pd.Series(a)
    num_series = pd.Series(a, index=['a', 'b', 'c', 'd'])  # with custom label
    print(f"series_1D_array: {num_series}")
    print(f'series_1D_array 1st value: {num_series["a"]}')  # access by label

    calories_dict = {"day1": 420, "day2": 380, "day3": 390}
    calories_series = pd.Series(calories_dict)
    print(f"calories_series: {calories_series}")
    pass


def pandas_dataframe():
    data = {
        "calories": [420, 380, 390],
        "duration": [50, 40, 45]
    }
    dataframe_food = pd.DataFrame(data)  # 2D labeled data structure with columns of potentially different types
    print(f"dataframe_food: \n{dataframe_food}")

    row_1 = dataframe_food.loc[0]  # access row by label
    print(f"dataframe_food row_1: \n{row_1}")

    row_1_and_2 = dataframe_food.loc[[0, 1]]  # access rows by label
    print(f"dataframe_food row_1_and_2: \n{row_1_and_2}")

    dataframe_food_named = pd.DataFrame(data, index=["day1", "day2", "day3"])  # with named index
    print(f"dataframe_food_named: \n{dataframe_food_named}")

    dataframe_food_csv = pd.read_csv("data/food_data.csv")  # read csv file
    print(f"dataframe_food_csv: \n{dataframe_food_csv}")

    dataframe_food_json = pd.read_json("data/food_data.json")  # read json file
    print(f"dataframe_food_json: \n{dataframe_food_json}")

    print(f"max_rows: {pd.options.display.max_rows}")  # Max rows displayed, if more than this then print will print header, first and last 5 rows
    pass


if __name__ == "__main__":
    pandas_usage()
