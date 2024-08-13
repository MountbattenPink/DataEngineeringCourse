from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

## Objective:
# Create a visual analysis of Airbnb listings in New York City using the dataset. This task will involve
# multiple visualizations that explore various aspects of the data, including neighborhood
# distribution, pricing trends, availability, room types, and review patterns.
# Task Breakdown:
# 1. Neighborhood Distribution of Listings
# o Plot: Create a bar plot to show the distribution of listings across different neighbourhood_group.
# o Details: Label each bar with the count of listings, use distinct colors for each neighborhood group, and add titles and axis labels.
csv_path = 'resources/cleaned_airbnb_data.csv'
data_df = pd.read_csv(csv_path, delimiter=',')
colors = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'])


neighbourhood_to_listing = data_df.groupby(['neighbourhood_group']).agg(listing=('price', 'count')).reset_index()
def create_bar_plot(keys_df, values_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = plt.bar(keys_df, values_df, color=colors, width=0.4)
    plt.xlabel("Neighbourhood Group")
    plt.ylabel("Counts of listings")
    plt.title("Airbnb Listings per Neighbourhood Group")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + 0.5, int(yval))
    plt.savefig(fname='resources/1_bar_listing_counts_per_neighbourhood_group.png')
    plt.close(fig)

keys_1 = neighbourhood_to_listing['neighbourhood_group'].to_list()
values_1 = neighbourhood_to_listing['listing'].to_list()

create_bar_plot(keys_1, values_1)

img = mpimg.imread('resources/1_bar_listing_counts_per_neighbourhood_group.png')
plt.imshow(img)
plt.show()

# 2. Price Distribution Across Neighborhoods
# o Plot: Generate a box plot to display the distribution of price within each neighbourhood_group.
# o Details: Use different colors for the box plots, highlight outliers, and add appropriate titles and axis labels.

neighbourhood_to_price = data_df.groupby('neighbourhood_group').agg(min_price=('price', 'min'), mean_price=('price', 'mean'), max_price=('price', 'max')).reset_index()


keys_2 = neighbourhood_to_price['neighbourhood_group'].to_list()
values_2 = neighbourhood_to_price[['min_price', 'mean_price', 'max_price']].values.tolist()

def create_bar_plot(keys_df, values_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = plt.boxplot(values_df, patch_artist=True, labels=keys_df)
    plt.xlabel("Neighbourhood Group")
    plt.ylabel("MIN/MEAN/MAX prices")
    plt.title("MIN/MEAN/MAX prices per Neighbourhood Group")
    for patch, color in zip(bp['boxes'], colors): patch.set_facecolor(color)
    for whisker in bp['whiskers']: whisker.set(color ='#8B008B', linewidth = 1.5, linestyle =":")
    for flier in bp['fliers']: flier.set(marker='o', color='#e7298a', alpha=0.7)
    plt.savefig(fname='resources/2_box_prices_per_neighbourhood_group.png')
    plt.close(fig)

create_bar_plot(keys_2, values_2)
img = mpimg.imread('resources/2_box_prices_per_neighbourhood_group.png')
plt.imshow(img)
plt.show()


# 3. Room Type vs. Availability
# o Plot: Create a grouped bar plot to show the average availability_365 for each room_type across the neighborhoods.
# o Details: Include error bars to indicate the standard deviation, use different colors
# for room types, and add titles and axis labels.
room_type_to_availability = data_df.groupby(['neighbourhood_group', 'room_type']).agg(
    mean=('availability_365', 'mean'),
    std=('availability_365', 'std')
).reset_index()


def create_grouped_bar_plot(df):
    fig, ax = plt.subplots(figsize=(10, 5))

    grouped = df.pivot_table(index='neighbourhood_group', columns='room_type', values=['mean', 'std'])
    neighbourhood_groups = grouped.index
    room_types = grouped.columns.levels[1]
    bar_width = 0.2
    x = np.arange(len(neighbourhood_groups))
    for i, room_type in enumerate(room_types):
        means = grouped['mean'][room_type]
        stds = grouped['std'][room_type]
        ax.bar(x + i * bar_width, means, bar_width, yerr=stds, capsize=5, label=room_type)

    ax.set_title('Mean and standard divergence availability_365 for neighbourhood group and room type')
    ax.set_xlabel('Neighbourhood Groups')
    ax.set_ylabel('availability_365')
    ax.legend(title='Room Type')
    plt.xticks(x, neighbourhood_groups)
    plt.tight_layout()
    plt.savefig('resources/3_grouped_bar_plot_availability_365.png')
    plt.close(fig)


create_grouped_bar_plot(room_type_to_availability)

img = mpimg.imread('resources/3_grouped_bar_plot_availability_365.png')
plt.imshow(img)
plt.show()



# 4. Correlation Between Price and Number of Reviews
# o Plot: Develop a scatter plot with price on the x-axis and number_of_reviews on the y-axis.
# o Details: Differentiate points by room_type using color or marker style, add a
# regression line to identify trends, and include a legend, titles, and axis labels.

price_to_reviews = data_df[data_df['price'].notna()][['number_of_reviews', 'price', 'room_type']]


def create_scatter_plot(df):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('price')
    ax.set_ylabel('number_of_reviews')
    ax.set_title('Correlation between number of reviews and price for different room types')

    unique_room_types = df['room_type'].unique()
    for i, room_type in enumerate(unique_room_types):
        sub_df = df[df['room_type'] == room_type]
        ax.scatter(sub_df['price'], sub_df['number_of_reviews'], color=colors[i], label=room_type)
        x = sub_df['price'].values.reshape(-1, 1)
        y = sub_df['number_of_reviews'].values
        model = LinearRegression().fit(x, y)
        predictions = model.predict(x)
        ax.plot(sub_df['price'], predictions, linestyle='-', color=colors[i], label=room_type)
    ax.legend(title='room_type')
    plt.savefig('resources/4_scatter_plot_reviews_to_price.png')
    plt.close(fig)

create_scatter_plot(price_to_reviews)

img = mpimg.imread('resources/4_scatter_plot_reviews_to_price.png')
plt.imshow(img)
plt.show()


# 5. Time Series Analysis of Reviews
# o Plot: Create a line plot to show the trend of number_of_reviews over time (last_review) for each neighbourhood_group.
# o Details: Use different colors for each neighborhood group, smooth the data with a
# rolling average, and add titles, axis labels, and a legend.

number_of_reviews_over_time = (data_df[data_df['last_review'].notna()][['neighbourhood_group', 'number_of_reviews', 'last_review']])

def create_line_plot(df):
    fig, ax = plt.subplots(figsize=(20, 20))
    df['last_review'] = pd.to_datetime(df['last_review'])
    ax.set_xlabel('last_review')
    ax.set_ylabel('number_of_reviews')
    ax.set_title('Trend of number_of_reviews over last_review for different neighbourhood_group')
    unique_groups = df['neighbourhood_group'].unique()
    for group in unique_groups:
        sub_df = df[df['neighbourhood_group'] == group]
        sub_df = sub_df.sort_values(by='last_review')
        sub_df['rolling_avg'] = sub_df['number_of_reviews'].rolling(window=2).mean()
        x = sub_df['last_review']
        y = sub_df['rolling_avg']
        ax.plot(x, y, label=group)

    ax.legend(title='Neighbourhood Group')
    plt.savefig('resources/5_line_plot_reviews_to_last_review.png')
    plt.close(fig)

create_line_plot(number_of_reviews_over_time)


img = mpimg.imread('resources/5_line_plot_reviews_to_last_review.png')
plt.imshow(img)
plt.show()


# 6. Price and Availability Heatmap
# o Plot: Generate a heatmap to visualize the relationship between price and
# availability_365 across different neighborhoods.
# o Details: Use a color gradient to represent the intensity of the relationship, label the
# axes, and include a color bar for reference.



# 7. Room Type and Review Count Analysis
# o Plot: Create a stacked bar plot to display the number_of_reviews for each
# room_type across the neighbourhood_group.
# o Details: Stack the bars by room type, use different colors for each room type, and add titles, axis labels, and a legend.


# Execution and Submission:
# • Python Script: Develop a single Python script using Matplotlib to generate all the
# visualizations. The script should be modular, with functions dedicated to each plot.
# • Data Preparation: Ensure the dataset is cleaned and pre-processed as necessary
# (e.g., handling missing values, categorizing data) before visualization.
#USED FILE FROM PREVIOUS PRACTICAL TASK: cleaned_airbnb_data.csv

# • Output: Save each plot as a separate image file (e.g., PNG) and also display them sequentially within the script.

# Deliverables:
# • A Python script (.py) that generates all the specified visualizations.
# • Individual image files for each plot (e.g., neighborhood_distribution.png, price_distribution.png)