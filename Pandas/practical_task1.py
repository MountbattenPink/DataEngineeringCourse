import numpy as np
import pandas as pd


##
# Practical Task 1: Basic Data Exploration and Cleaning with Pandas
# Objective:
# Introduce basic data exploration and data cleaning techniques using the New York City Airbnb Open Data dataset.
# Requirements:

# Output Function:
# o Implement a separate function named print_dataframe_info that takes the DataFrame
# and an optional message as input, prints the message, and displays basic information
# about the DataFrame (e.g., number of entries, missing values).
# o Use this function to output the state of the DataFrame before and after cleaning.
def print_dataframe_info(df, msg=''):
    print('\n' + msg)
    print(df.info())


def print_dataframe(df, msg=''):
    print('\n' + msg)
    print(df.to_string())


# 1. Data Loading and Initial Inspection:
# o Load the New York City Airbnb Open Data dataset into DataFrame.

csv_path = 'resources/AB_NYC_2019.csv'
data_df = pd.read_csv(csv_path, delimiter=',')
copy_for_tests = data_df.copy(deep=True)

# o Inspect the first few rows of the dataset using the head() method to understand its structure.

first_rows = data_df.head(10)
print('DataFrame structure: ', data_df.columns)
print_dataframe_info(first_rows, 'First 10 rows info: ')

# o Retrieve basic information about the dataset, such as the number of entries, column
# data types, and memory usage, using the info() method.
print_dataframe_info(data_df, 'Basic dataframe info: ')

# 2. Handling Missing Values:
# o Identify columns with missing values and count the number of missing entries per column.
print_dataframe(data_df.isna().sum().sort_values(ascending=False), 'Columns with numbers of NA values:')

# o Handle missing values in the name, host_name, and last_review columns:
# ▪ For name and host_name, fill missing values with the string "Unknown".
data_df.loc[data_df['name'].isna(), 'name'] = 'Unknown'
data_df.loc[data_df['host_name'].isna(), 'host_name'] = 'Unknown'
print_dataframe(data_df.isna().sum().sort_values(ascending=False),
                'Input data after cleaning "name" and "host_name" empty values:')

# ▪ For last_review, fill missing values with a special value “NaT". “NaT" stands for Not a Time.
data_df.loc[data_df['last_review'].isna(), 'last_review'] = pd.NaT
print_dataframe(data_df.isna().sum().sort_values(ascending=False),
                'Input data after cleaning "last_review" empty values:')

# 3. Data Transformation:
# o Categorize Listings by Price Range:
# Create a new column price_category that categorizes listings into different price ranges, such as Low, Medium, High,
# based on defined thresholds (e.g., Low: price < $100, Medium: $100 <= price < $300, High: price >= $300).
data_df['price_category'] = pd.cut(data_df['price'],
                                   bins=[float('-inf'), 100, 300, float('inf')],
                                   right=False,
                                   labels=['Low', 'Medium', 'High'])
print_dataframe_info(data_df, 'Input data with new "price_category" column: ')

# o Create a length_of_stay_category column:
# Categorize listings based on their minimum_nights into short-term, medium-term, and long-term stays.
# For example, short-term might be minimum_nights <= 3, medium-term
# minimum_nights between 4 and 14, and long-term minimum_nights > 14.
data_df['length_of_stay_category'] = pd.cut(data_df['minimum_nights'], include_lowest=True,
                                            bins=[0, 3, 14, float('inf')],
                                            labels=['short-term', 'medium-term', 'long-term']
                                            )
print_dataframe_info(data_df, 'Input data with new "length_of_stay_category" column: ')

def generate_test_data_get_length_of_stay_category(minimum_nights):
    if minimum_nights <= 3:
        return 'short-term'
    elif minimum_nights > 14:
        return 'long-term'
    else:
        return 'medium-term'


def generate_test_data_get_price_category(price):
    if price < 100:
        return 'Low'
    elif price >= 300:
        return 'High'
    else:
        return 'Medium'


def generate_test_data_clear_string_field(field):
    if pd.isna(field):
        return 'Unknown'
    else:
        return field


def generate_test_data_clear_date_time_field(field):
    if pd.isna(field):
        return pd.NaT
    else:
        return field


copy_for_tests['name'] = copy_for_tests['name'].apply(generate_test_data_clear_string_field)
copy_for_tests['last_review'] = copy_for_tests['last_review'].apply(generate_test_data_clear_date_time_field)
copy_for_tests['host_name'] = copy_for_tests['host_name'].apply(generate_test_data_clear_string_field)
copy_for_tests['price_category'] = copy_for_tests['price'].apply(generate_test_data_get_price_category).astype('category')
copy_for_tests['price_category'] = pd.Categorical(copy_for_tests['price_category'],
                                                  categories=['Low', 'Medium', 'High'], ordered=True)
copy_for_tests['length_of_stay_category'] = copy_for_tests['minimum_nights'].apply(generate_test_data_get_length_of_stay_category)
copy_for_tests['length_of_stay_category'] = pd.Categorical(copy_for_tests['length_of_stay_category'],
                                                           categories=['short-term', 'medium-term', 'long-term'],
                                                           ordered=True)

pd.testing.assert_frame_equal(data_df, copy_for_tests)

# o Ensure that the dataset has no missing values in critical columns (name, host_name, last_review).
np.testing.assert_equal(0, data_df['name'].isna().sum())
np.testing.assert_equal(0, data_df['host_name'].isna().sum())
np.testing.assert_equal(0, len(data_df[data_df['last_review'] == '']))
# o Confirm that all price values are greater than 0. If you find rows with price equal to 0, remove them.
data_df.loc[data_df['price'] == 0, 'price'] = pd.NA
np.testing.assert_equal(0, len(data_df[data_df['price'] <= 0]))



# 4. Data Validation:
# o Verify that the data transformations and cleaning steps were successful by reinspecting the DataFrame.
# Test the script to ensure that all data loading, cleaning, and transformation operations are
# executed correctly. Validate that the cleaned dataset is ready for further analysis.
# TESTS ARE ADDED IN EACH CHAPTER


# Save the cleaned dataset as a new CSV file named cleaned_airbnb_data.csv for use in subsequent tasks.
data_df.to_csv(path_or_buf='resources/cleaned_airbnb_data.csv', index=False)
