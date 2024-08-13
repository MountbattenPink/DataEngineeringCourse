import numpy as np
import pandas as pd
import calendar

## Practical Task 3: Advanced Data Manipulation, Descriptive Statistics, and Time Series
# Analysis with Pandas
# Objective: Enhance data manipulation skills while generating actionable business insights by
# performing advanced transformations, descriptive statistics, and time series analysis on the New York City Airbnb Open Data dataset.
# Requirements:

# 4. Output Function:
# o Implement a separate function named print_analysis_results that takes the
# DataFrame or Series and an optional message as input, prints the message, and
# displays the results of your analysis. This function should be used to clearly present
# the findings from your descriptive statistics and time series analysis.
def print_analysis_results(input_data, msg=''):
    print('\n', msg)
    print(input_data.to_string())


# 1. Advanced Data Manipulation:
csv_path = 'resources/cleaned_airbnb_data.csv'
data_df = pd.read_csv(csv_path)



# o Analyze Pricing Trends Across Neighborhoods and Room Types:
# ▪ Use the pivot_table function to create a detailed summary that reveals the
# average price for different combinations of neighbourhood_group and
# room_type. This analysis will help identify high-demand areas and optimize
# pricing strategies across various types of accommodations (e.g., Entire home/apt vs. Private room).

pivot_table = pd.pivot_table(
    data_df, values='price', index='neighbourhood_group', columns='room_type', aggfunc='mean', fill_value=0
)

print_analysis_results(pivot_table, 'Pivot table: ')


# o Prepare Data for In-Depth Metric Analysis:
# ▪ Transform the dataset from a wide format to a long format using the melt
# function. This restructuring facilitates more flexible and detailed analysis of
# key metrics like price and minimum_nights, enabling the identification of
# trends, outliers, and correlations.
melt_table = pd.melt(data_df, id_vars=['neighbourhood_group', 'room_type'], value_vars=['price', 'minimum_nights'],
                     var_name='trends', value_name='value_name')


# o Classify Listings by Availability:
# ▪ Create a new column availability_status using the apply function,
# classifying each listing into one of three categories based on the availability_365 column:
# ▪ "Rarely Available": Listings with fewer than 50 days of availability in
# a year.
# ▪ "Occasionally Available": Listings with availability between 50 and
# 200 days.
# ▪ "Highly Available": Listings with more than 200 days of availability.
def classify_availability(availability_365):
    if (availability_365<50):
        return 'Rarely Available'
    elif (availability_365>200):
        return 'Highly Available'
    else:
        return 'Occasionally Available'

data_df['availability_status'] = data_df['availability_365'].apply(classify_availability)


# ▪ Analyze trends and patterns using the new availability_status column, and
# investigate potential correlations between availability and other key
# variables like price, number_of_reviews, and neighbourhood_group to
# uncover insights that could inform marketing and operational strategies.
# 2. Descriptive Statistics:
# o Perform basic descriptive statistics (e.g., mean, median, standard deviation) on
# numeric columns such as price, minimum_nights, and number_of_reviews to
# summarize the dataset's central tendencies and variability, which is crucial for
# understanding overall market dynamics.

descriptive_statistics = data_df[['price', 'minimum_nights', 'number_of_reviews']].agg(['mean', 'median', 'std'])

print_analysis_results(descriptive_statistics, 'Descriptive statistics: ')


# 3. Time Series Analysis:
# o Convert and Index Time Data:
# ▪ Convert the last_review column to a datetime object and set it as the index
# of the DataFrame to facilitate time-based analyses.
data_df['last_review'] = pd.to_datetime(data_df['last_review'])
data_df.set_index('last_review')


# o Identify Monthly Trends:
# ▪ Resample the data to observe monthly trends in the number of reviews and
# average prices, providing insights into how demand and pricing fluctuate over time.
monthly_trends = data_df.resample('M', on='last_review').agg({'price': 'mean'})

print_analysis_results(monthly_trends, 'Monthly price trends: ')

# o Analyze Seasonal Patterns:
# ▪ Group the data by month to calculate monthly averages and analyze
# seasonal patterns, enabling better forecasting and strategic planning around peak periods.
data_df['month'] = pd.DatetimeIndex(data_df['last_review']).month
seasonal_trends = data_df.groupby(['month']).agg({'price': 'mean', 'number_of_reviews': 'mean'})

print_analysis_results(seasonal_trends, 'Seasonal trends:')


# 5. Execution and Verification:
# o Test the script to ensure that all data manipulation, statistical, and time series
# operations are executed correctly.
# o Validate that the results provide meaningful insights, are accurate, and are ready for
# visualization or further analysis, ensuring the data-driven decisions are wellsupported.

# Deliverables:
# • A Python script (.py file) containing all the functions and code necessary to perform the
# advanced data manipulation, descriptive statistics, and time series analysis, and to output
# the results using the print_analysis_results function.

# • Save the results of your time series analysis as a new CSV file named time_series_airbnb_data.csv
seasonal_trends.to_csv(path_or_buf='resources/time_series_airbnb_data.csv', index=False)