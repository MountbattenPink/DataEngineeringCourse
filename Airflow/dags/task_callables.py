import os
from io import StringIO
import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook
import logging

logger = logging.getLogger("airflow.task")


def ingest_data(csv_path, **kwargs):
    if not os.path.exists(csv_path):
        raise IOError(f"Cannot find file '{csv_path}'")
    elif not os.access(csv_path, os.R_OK):
        raise IOError(f"Cannot open file '{csv_path}'")
    else:
        try:
            data_df = pd.read_csv(csv_path, delimiter=',')
            data_df_csv = data_df.to_csv(index=False)
            kwargs['ti'].xcom_push(key='ingested_data', value=data_df_csv)
        except Exception as e:
            raise IOError(f"Failed to read file '{csv_path}'")



def transform(output_file_path, **kwargs):
    df_csv = kwargs['ti'].xcom_pull(key='ingested_data', task_ids='ingest_data_task')

    data_df = pd.read_csv(StringIO(df_csv))

    data_df = data_df.loc[data_df['price'] > 0]

    data_df['last_review'] = pd.to_datetime(data_df['last_review'])
    earliest_date = data_df['last_review'].min()
    data_df.loc[data_df['last_review'].isna(), 'last_review'] = earliest_date

    data_df.loc[data_df['reviews_per_month'].isna(), 'reviews_per_month'] = 0

    data_df = data_df.loc[(data_df['latitude'].notna() & data_df['longitude'].notna())]
    data_df.to_csv(path_or_buf=output_file_path, index=False)
    # for further quality check
    kwargs['ti'].xcom_push(key='transformed_data_length', value=len(data_df))

    # since we already have this dataframe in memory, let's also optimize data and create sql insert script
    # for the further loading to db:
    data_df['name'] = data_df['name'].astype(str)
    data_df['host_name'] = data_df['host_name'].astype(str)
    data_df['neighbourhood_group'] = data_df['neighbourhood_group'].astype(str)
    data_df['neighbourhood'] = data_df['neighbourhood'].astype(str)

    inserts = []
    for _, row in data_df.iterrows():
        name = row['name'].replace("'", "''").replace("$", "\$")
        host_name = row['host_name'].replace("'", "''").replace("$", "\$")
        neighbourhood_group = row['neighbourhood_group'].replace("'", "''").replace("$", "\$")
        neighbourhood = row['neighbourhood'].replace("'", "''").replace("$", "\$")

        inserts.append(f"""BEGIN
        insert into airbnb_listings
        (id, name, host_id, host_name, neighbourhood_group, neighbourhood, latitude, longitude,
         room_type, price, minimum_nights, number_of_reviews, last_review, reviews_per_month,
         calculated_host_listings_count, availability_365) 
        values 
        ('{row['id']}','{name}', {row['host_id']}, '{host_name}', '{neighbourhood_group}',
        '{neighbourhood}', TRUNC({row['latitude']},6),TRUNC({row['longitude']},6),'{row['room_type']}',
        {row['price']},{row['minimum_nights']},{row['number_of_reviews']},'{row['last_review']}',
        TRUNC({row['reviews_per_month']},2), {row['calculated_host_listings_count']},{row['availability_365']});
        exception WHEN OTHERS THEN RAISE NOTICE 'Error occurred for {row['id']}'; END; """)
    content = ''.join(inserts)
    content = "DO $$ BEGIN\n" + content + "\nEND $$;"

    with open("sql/nyc_airbnb_inserts.sql", 'w') as file:
        file.write(content)


def check_quality(db_connection_id, **kwargs):
    expected_row_number = kwargs['ti'].xcom_pull(key='transformed_data_length', task_ids='transform_data_task')
    pg_hook = PostgresHook(postgres_conn_id=db_connection_id)
    row_number = pg_hook.get_records("SELECT COUNT(*) FROM airbnb_listings")[0][0]
    if len != expected_row_number:
        raise ValueError(f"Error: number of rows in DB {row_number} not as expected {expected_row_number}")
    elif pg_hook.get_records("""
            select count(price)
            from airbnb_listings
            where price is null or minimum_nights is null or availability_365 is null
        """)[0][0]  > 0:
        raise ValueError(f"Error: some prices/minimum_nights/availability_365 are empty")
    return 'success_task'

def log_error():
    raise ValueError("Error found during data quality check")

def log_success_execution():
    return 'succeed'


# 8. Implement Error Handling and Logging Failures to a File:
# o Configure Airflow to log task failures to a local file instead of sending email notifications.
# o Implement a failure callback function in your code that writes failure details to a log file.
def log_error_to_file(context):
    task_instance = context.get('task_instance')
    error_msg = str(context.get('exception')) if context.get('exception') else ''
    logging.error(f"Task {task_instance} failed with exception {error_msg}")
