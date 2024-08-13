import numpy as np
import pandas as pd


## Practical Task 2: Data Selection, Filtering, and Aggregation with Pandas
# Objective:
# Develop skills in selecting, filtering, and aggregating data within the New York City Airbnb Open
# Data dataset.
# Requirements:

# 4. Output:
# o Implement a separate function named print_grouped_data that takes the grouped
# DataFrame and an optional message as input, prints the message, and displays the grouped data.
# o Output the results of your aggregations and rankings.

def print_grouped_data(groupedDF, msg=''):
    print('\n', msg)
    print(groupedDF.to_string())


# 1. Data Selection and Filtering:
csv_path = 'resources/cleaned_airbnb_data.csv'
data_df = pd.read_csv(csv_path)


# o Use .iloc and .loc to select specific rows and columns based on both position and labels.
# o Filter the dataset to include only listings in specific neighborhoods (e.g., Manhattan,Brooklyn).
def filter_by_neighbourhood(df, neighbourhoods):
    return df.loc[df['neighbourhood_group'].isin(neighbourhoods)]


def generate_test_data_count_by_neighbourhood(df, neighbourhoods):
    res = 0
    for index, row in df.iterrows():
        if (neighbourhoods.__contains__(row['neighbourhood_group'])):
            res += 1
    return res

#tests
np.testing.assert_equal(generate_test_data_count_by_neighbourhood(data_df, ['Manhattan']),
                        len(filter_by_neighbourhood(data_df, ['Manhattan'])))
np.testing.assert_equal(generate_test_data_count_by_neighbourhood(data_df, ['Brooklyn']),
                        len(filter_by_neighbourhood(data_df, ['Brooklyn'])))


data_df = filter_by_neighbourhood(data_df, ['Manhattan', 'Brooklyn'])


# o Further filter the dataset to include only listings with a price greater than $100 and a number_of_reviews greater than 10.
def filter_by_price_and_reviews(df, min_price=100, min_reviews=10):
    return df.loc[(df['price'] > min_price) & (df['number_of_reviews'] > min_reviews)]


def generate_test_data_filter_by_price_and_reviews(df, price, reviews):
    filtered_rows = []
    for index, row in df.iterrows():
        if (row['number_of_reviews'] > reviews) and (row['price'] > price):
            filtered_rows.append(row)
    filtered_df = pd.DataFrame(filtered_rows, columns=df.columns).astype(df.dtypes)
    return filtered_df


#tests
pd.testing.assert_frame_equal(generate_test_data_filter_by_price_and_reviews(data_df, 100, 10),
                              filter_by_price_and_reviews(data_df))
pd.testing.assert_frame_equal(generate_test_data_filter_by_price_and_reviews(data_df, 200, 5),
                              filter_by_price_and_reviews(data_df, 200, 5))


data_df = filter_by_price_and_reviews(data_df)


# o Select columns of interest such as neighbourhood_group, price, minimum_nights,
# number_of_reviews, price_category and availability_365 for further analysis.
data_df = data_df[
    ['neighbourhood_group', 'price', 'minimum_nights', 'number_of_reviews', 'price_category', 'availability_365']]

np.testing.assert_array_equal(
    np.array(data_df.columns),
    np.array(
        ['neighbourhood_group', 'price', 'minimum_nights', 'number_of_reviews', 'price_category', 'availability_365'])
)


# 2. Aggregation and Grouping:
# o Group the filtered dataset by neighbourhood_group and price_category to calculate aggregate statistics:
# ▪ Calculate the average price and minimum_nights for each group.
# ▪ Compute the average number_of_reviews and availability_365 for each group to understand typical
# review counts and availability within each neighborhood and price category.


def aggregate_df(df):
    return df.groupby(['neighbourhood_group', 'price_category']).agg(
    mean_price=('price', 'mean'),
    mean_minimum_nights=('minimum_nights', 'mean'),
    mean_number_of_reviews=('number_of_reviews', 'mean'),
    mean_availability_365=('availability_365', 'mean'),
    listing_count=('price', 'count')
).reset_index()


data_df = aggregate_df(data_df)
print_grouped_data(data_df,
                   'Filtered, grouped and with average values:')


#tests
test_data = pd.DataFrame({
    'neighbourhood_group': ['Brooklyn', 'Brooklyn', 'Manhattan', 'Manhattan'],
    'price_category': ['High', 'Medium', 'High', 'Medium'],
    'mean_price': [433.168317, 160.894980, 515.126761, 172.955793],
    'mean_minimum_nights': [4.884488, 4.786880, 5.429577, 5.148409],
    'mean_number_of_reviews': [48.864686, 61.624577, 44.825352, 55.138207],
    'mean_availability_365': [185.900990, 141.537111, 176.502817, 123.767306],
    'listing_count': [303, 3247, 710, 4117]
})

for column in test_data.columns.to_list():
    pd.testing.assert_series_equal(
        test_data.sort_values(column).reset_index()[column],
        data_df.sort_values(column).reset_index()[column]
    )



# 3. Data Sorting and Ranking:
# o Sort the data by price in descending order and by number_of_reviews in ascending order.
data_df = data_df.sort_values(['mean_price', 'mean_number_of_reviews'], ascending=[False, True])
print_grouped_data(data_df,
                   'Filtered, grouped and with average values. sorted by price and number of reviews:')



#tests
test_data.sort_values(['mean_price', 'mean_number_of_reviews'], ascending=[False, True])
for column in test_data.columns.to_list():
    pd.testing.assert_series_equal(
        test_data.sort_values(column).reset_index()[column],
        data_df.sort_values(column).reset_index()[column]
    )



# o Create a ranking of neighborhoods based on the total number of listings and the average price.
data_df['rank_listings'] = data_df['listing_count'].rank(ascending=False, method='min')
data_df['rank_price'] = data_df['mean_price'].rank(ascending=False, method='min')
print_grouped_data(data_df, 'Ranked by listings and mean price:')


#tests
np.testing.assert_equal(data_df[data_df['rank_listings'] == 1]['neighbourhood_group'].iloc[0], 'Manhattan')
np.testing.assert_equal(data_df[data_df['rank_listings'] == 1]['price_category'].iloc[0], 'Medium')
np.testing.assert_equal(data_df[data_df['rank_listings'] == 3]['neighbourhood_group'].iloc[0], 'Manhattan')
np.testing.assert_equal(data_df[data_df['rank_listings'] == 3]['price_category'].iloc[0], 'High')
np.testing.assert_equal(data_df[data_df['rank_listings'] == 2]['neighbourhood_group'].iloc[0], 'Brooklyn')
np.testing.assert_equal(data_df[data_df['rank_listings'] == 2]['price_category'].iloc[0], 'Medium')
np.testing.assert_equal(data_df[data_df['rank_listings'] == 4]['neighbourhood_group'].iloc[0], 'Brooklyn')
np.testing.assert_equal(data_df[data_df['rank_listings'] == 4]['price_category'].iloc[0], 'High')

np.testing.assert_equal(data_df[data_df['rank_price'] == 3]['neighbourhood_group'].iloc[0], 'Manhattan')
np.testing.assert_equal(data_df[data_df['rank_price'] == 3]['price_category'].iloc[0], 'Medium')
np.testing.assert_equal(data_df[data_df['rank_price'] == 1]['neighbourhood_group'].iloc[0], 'Manhattan')
np.testing.assert_equal(data_df[data_df['rank_price'] == 1]['price_category'].iloc[0], 'High')
np.testing.assert_equal(data_df[data_df['rank_price'] == 4]['neighbourhood_group'].iloc[0], 'Brooklyn')
np.testing.assert_equal(data_df[data_df['rank_price'] == 4]['price_category'].iloc[0], 'Medium')
np.testing.assert_equal(data_df[data_df['rank_price'] == 2]['neighbourhood_group'].iloc[0], 'Brooklyn')
np.testing.assert_equal(data_df[data_df['rank_price'] == 2]['price_category'].iloc[0], 'High')


# 5. Execution and Verification:
# o Test the script to ensure that all selection, filtering, and aggregation operations are executed correctly.
# o Validate that the filtered and grouped dataset is accurate and provides meaningful insights.
# TESTS ARE ADDED IN EACH CHAPTER

# Save the aggregated data as a new CSV file named aggregated_airbnb_data.csv
data_df.to_csv(path_or_buf='resources/aggregated_airbnb_data.csv', index=False)
