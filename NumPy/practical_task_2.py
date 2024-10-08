import numpy as np


# Practical Task 2: Analyzing and Visualizing E-Commerce Transactions with NumPy
# Objective:
# Develop a set of Python functions using NumPy to manipulate and analyze a simulated ecommerce dataset.

# Output Function: Implement a separate function named print_array that takes an array and an optional
# message as input, and prints them to the console. This function will be used to display
# the state of the array, maintaining separation from the manipulation logic.
def print_array(arr, msg=""):
    print(msg, arr)


# Requirements:
# 1. Array Creation:
# Generate a multi-dimensional NumPy array with predefined values to simulate ecommerce transactions. The array should include transaction_id, user_id,
# product_id, quantity, price, and timestamp.
dType = np.dtype(
    [('transaction_id', 'S50'), ('user_id', 'S50'), ('product_id', 'S50'), ('quantity', np.int32), ('price', np.float64),
     ('timestamp', 'datetime64[s]')])

data = np.empty(0, dtype=dType)


# this function generates data array. you can see resulted array from the output (print_array function was used)
# additionally, it gets all the numbers for further test assertions (using simple math, sets and maps, no numpy)
def generateDataSetAndDataForTests(n):
    global data
    print("np.array([")
    total_revenue = 0
    unique_user_ids = set()
    products_purchases = {}
    transactions_per_user = {}
    product_revenues = {}

    for i in range(n):
        tx_id = f'tx-{np.random.randint(1, 1000)}'
        user_id = f'user-{np.random.randint(1, 30)}'
        product_id = f'product-{np.random.randint(1, 50)}'
        # to have for sure some records for future exercise with masking
        quantity = 0 if (i % 10 == 0) else np.random.randint(0, 1000)
        # since it's price, it cannot have >2 digits after comma
        price = np.round(np.random.uniform(20, 1000), 2)
        now = np.datetime64('now')
        record_timestamp = now - np.timedelta64(np.random.randint(0, 10000), 'h')

        total_revenue += quantity * price
        unique_user_ids.add(user_id)

        if product_id not in products_purchases:
            products_purchases[product_id] = 0
        products_purchases[product_id] += quantity

        if user_id not in transactions_per_user:
            transactions_per_user[user_id] = 0
        transactions_per_user[user_id] += 1

        if product_id not in product_revenues:
            product_revenues[product_id] = 0
        product_revenues[product_id] += quantity*price

        record = np.array([(tx_id, user_id, product_id, quantity, price, record_timestamp)], dtype=dType)
        data = np.concatenate((data, record))

    print_array(data, '\nGenerated array:')

    return (total_revenue, len(unique_user_ids), max(products_purchases, key=products_purchases.get), transactions_per_user, sorted(product_revenues, key=product_revenues.get, reverse=True)[:5])


expected_test_values = generateDataSetAndDataForTests(50)


# 2. Data Analysis Functions:
# o Total Revenue Function: Create a function to calculate the total revenue generated by multiplying quantity and price, and summing the result.

# argument added for function reusing in 4th part of task
def get_total_revenue(arr=data):
    return np.sum(arr['quantity'] * arr['price'])

total_revenue = get_total_revenue()
print("\nTotal revenue:", total_revenue)
np.testing.assert_almost_equal(total_revenue, expected_test_values[0])


#Unique Users Function: Develop a function to determine the number of unique users who made transactions.
def get_unique_users():
    return np.unique(data['user_id']).size


unique_users_count = get_unique_users()
print("\nUnique users: ", unique_users_count)
np.testing.assert_equal(unique_users_count, expected_test_values[1])


# o Most Purchased Product Function: Implement a function to identify the most purchased product based on the quantity sold.
def get_most_purchased_product():
    product_ids = np.copy(data['product_id'])
    quantities = np.copy(data['quantity'])
    unique_ids, inverse_indices = np.unique(product_ids, return_inverse=True)
    total_quantities = np.zeros(len(unique_ids), dtype=np.int32)
    np.add.at(total_quantities, inverse_indices, quantities)
    result_dtype = np.dtype([('product_id', 'S50'), ('total_quantity', np.int32)])
    products_with_total_quan = np.array(list(zip(unique_ids, total_quantities)), dtype=result_dtype)
    max_quantity_index = np.argmax(products_with_total_quan['total_quantity'])
    return products_with_total_quan[max_quantity_index][0].decode('utf-8')


most_purchased_product = get_most_purchased_product()
print_array(most_purchased_product, "\nMost purchased product: ")
np.testing.assert_equal(most_purchased_product, expected_test_values[2])


# o Type Casting and Checking Functions:
# ▪ Create a function to convert the price array from float to integer.

#since the price are float, it makes sense to convert it to coins/cents too (x100)
def convert_price_to_integer_numpy():
    prices = np.copy(data['price'])
    prices_int = np.int64(prices*100)
    return prices_int

def convert_price_to_integer_pure_python():
    prices = np.copy(data['price'])
    for i in range(len(prices)):
        prices[i] = (int)(prices[i]*100)
    return prices

converted_prices_to_integer_and_to_cents_numpy = convert_price_to_integer_numpy()
print_array(converted_prices_to_integer_and_to_cents_numpy, "\nPrices after convertion from float to int and to cents:")
np.testing.assert_equal(converted_prices_to_integer_and_to_cents_numpy, convert_price_to_integer_pure_python())

# Develop a function to check the data types of each column in the array.
def check_column_types():
    for name in data.dtype.names:
        print("Column type: ", name, data.dtype.fields.get(name))

print("\nChecking column types:")
check_column_types()


# 3. Array Manipulation Functions:
# o Product Quantity Array Function: Create a function that returns a new array with only the product_id and quantity columns.
def get_product_and_quantity():
    return data[['product_id', 'quantity']]

print_array(get_product_and_quantity(), "\nOnly 'product_id' and 'quantity' columns")



# o User Transaction Count Function: Implement a function to generate an array of transaction counts per user.
def transactions_per_user():
    sorted_data = np.sort(data, order='user_id')
    unique_user_ids, start_indices = np.unique(sorted_data['user_id'], return_index=True)
    groups = {user_id.decode('utf-8'): len(sorted_data[sorted_data['user_id'] == user_id]) for user_id in unique_user_ids}
    return groups

transactions_count_per_user = transactions_per_user()
print_array(transactions_count_per_user, '\nTransactions per user')
#testing value for each user
for user in transactions_count_per_user:
    np.testing.assert_equal(transactions_count_per_user[user], expected_test_values[3][user])



# o Masked Array Function: Create a function to generate a masked array that hides transactions where the quantity is zero.
def mask_zero_quantities():
    mask = (data['quantity'] == 0)
    masked = np.ma.masked_array(data, mask=[mask])
    return masked

masked_array = mask_zero_quantities()
print_array(masked_array, "\nMasked all zero quantities:")
for i in range(len(masked_array)):
    np.testing.assert_array_equal(masked_array[i], data[i])




# 4. Arithmetic and Comparison Functions:
# o Price Increase Function: Develop a function to increase all prices by a certain percentage (e.g., 5% increase).
def increase_all_prices(percent):
    increased_data = np.copy(data)
    increased_data['price'] = increased_data['price'] * (1 + percent/100)
    return increased_data

increased_array = increase_all_prices(5)
print_array(increased_array, "\nIncreased prices by 5%: ")
for i in range(len(data)):
    np.testing.assert_equal(increased_array[i]['price'], data[i]['price'] * 1.05)





# o Filter Transactions Function: Implement a function to filter transactions to only include those with a quantity greater than 1.
def filter_quantities_g_1():
    return data[data['quantity'] > 1]

filtered_quantities = filter_quantities_g_1()
print_array(filtered_quantities, "\nFiltered out all records with quantities <=1")
#added for testing
def filter_quantities_le_1():
    return data[data['quantity'] <= 1]
filtered_out_quantities = filter_quantities_le_1()
np.testing.assert_equal(len(filtered_quantities)+len(filtered_out_quantities), len(data))


# o Revenue Comparison Function: Create a function to compare the revenue from two different time periods.
def compare_revenue(start1, end1, start2, end2):
    return (get_total_revenue(data[(data['timestamp'] >= start1) & (data['timestamp'] <= end1)]) -
            get_total_revenue(data[(data['timestamp'] >= start2) & (data['timestamp'] <= end2)]))

now = np.datetime64('now')
start1 = now - np.timedelta64(np.random.randint(1, 1000), 'h')
start2 = start1 - np.timedelta64(np.random.randint(1, 1000), 'h')

np.testing.assert_equal(compare_revenue(start1, now, start1, now), 0)
np.testing.assert_(compare_revenue(start1, now, start2, now) <= 0)
np.testing.assert_(compare_revenue(start2, now, start1, now) >= 0)




# 5. Indexing and Slicing Functions:
# o User Transactions Function: Create a function to extract all transactions for a specific user.
def transactions_by_user_id(user_id):
    return data[data['user_id'] == user_id.encode('utf-8')]

for u in transactions_count_per_user:
    np.testing.assert_equal(len(transactions_by_user_id(u)), transactions_count_per_user[u])


# o Date Range Slicing Function: Develop a function to slice the dataset to include only transactions within a specific date range.
def slice_dataset_within_date_range(start_date, end_date):
    start = np.datetime64(start_date + 'T00:00:00')
    end = np.datetime64(end_date + 'T23:59:59')
    return data[(data['timestamp'] >= start) & (data['timestamp'] <= end)]

print_array(slice_dataset_within_date_range('2024-08-01', '2024-12-31'), "\nAll records within date range: ")

np.testing.assert_equal(len(data), len(slice_dataset_within_date_range('1970-01-01', '2024-12-31')))
np.testing.assert_equal(len(data),
                        len(slice_dataset_within_date_range('1970-01-01', '2024-01-31')) +
                        len(slice_dataset_within_date_range('2024-02-01', '2024-12-31')))


# o Top Products Function: Implement a function using advanced indexing to retrieve transactions of the top 5 products by revenue.
def top_5_product_records_by_revenue():
    product_ids = np.copy(data['product_id'])
    revenues = np.copy(data['quantity']*data['price'])
    unique_ids, inverse_indices = np.unique(product_ids, return_inverse=True)
    total_revenues = np.zeros(len(unique_ids), dtype=np.float64)
    np.add.at(total_revenues, inverse_indices, revenues)
    result_dtype = np.dtype([('product_id', 'S50'), ('total_revenue', np.float64)])
    products_with_total_revenue = np.array(list(zip(unique_ids, total_revenues)), dtype=result_dtype)
    top_products = np.sort(products_with_total_revenue, order='total_revenue')
    top_products = top_products[(len(top_products)-5): len(top_products)]['product_id']
    return data[np.isin(data['product_id'], top_products)], top_products

top_5_product_records = top_5_product_records_by_revenue()
print_array(top_5_product_records, "\nTop 5 product transactions: ")
np.testing.assert_array_equal(top_5_product_records[1].sort(), expected_test_values[4].sort())