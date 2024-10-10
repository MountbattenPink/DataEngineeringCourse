import boto3
import logging
import psycopg2
import pandas as pd
from io import StringIO
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

#init S3 client
s3 = boto3.client('s3')

#DB connection config
database = 'dev'
user = 'admin'
password = 'Password1234'
host = 'default-workgroup.700128968765.eu-west-1.redshift-serverless.amazonaws.com'
redshift_port = '5439'

#raw table for storing the original unprocessed data
redshift_raw_table = 'raw'

#transformed table for storing cleaned data
redshift_transformed_table = 'transformed'

#role for lambda function, has access to s3, cloudwatch, and redshift data api
iam_role_arn = 'arn:aws:iam::700128968765:role/lambda-s3-trigger-role'
s3_bucket = 'data-engineering-olsh'
conn_str = f"dbname={database} host={host} port={redshift_port} user={user} password={password}"

"""
loads the raw data to the Redshift raw table, transforms/cleans it,
and stores the transformed data in the transformed table.
"""
def lambda_handler(event, context):
    for record in event['Records']:
        #get the S3 csv file path
        s3_key = record['s3']['object']['key']
        logger.info(f"New raw data: {s3_key}")

        s3_path = f's3://{s3_bucket}/{s3_key}'

        try:
            #get db connection
            conn = psycopg2.connect(conn_str)
            cursor = conn.cursor()

            #load raw data from S3 to raw table
            load_data_to_raw_redshift(s3_path, conn, cursor)

            #take csv data from S3 for transformation
            csv_data = s3.get_object(Bucket=s3_bucket, Key=s3_key)['Body'].read().decode('utf-8')

            #make dataframe from csv
            df = pd.read_csv(StringIO(csv_data))
            df = transform_data(df)

            #load transformed data to transformed table
            write_to_redshift(df, conn, cursor)
        except Exception as e:
            logger.error(f"Excepction occured while handling data: {e}")
            raise
        finally:
            cursor.close()
            conn.close()


    return {
        'statusCode': 200,
        'body': "Successfully handled new data"
    }



def load_data_to_raw_redshift(s3_path):
    """
    load raw data from S3 into the Redshift raw table with COPY command.
    """
    try:

        #COPY command string
        copy_sql =
        f"""
        COPY {redshift_raw_table}
        FROM '{s3_path}'
        IAM_ROLE '{iam_role_arn}'
        CSV
        IGNOREHEADER 1;
        """

        cursor.execute(copy_sql)
        conn.commit()
        logger.info(f"Successfully loaded raw data")

    except Exception as e:
        logger.error(f"Excepction occured while loading raw data: {e}")
        raise


"""
transforms and cleans raw data
"""
def transform_data(df):
    #filter out rows where price is 0 or negative
    df = df[df['price'] > 0]
    #filter out rows with missing latitude or longitude
    df.dropna(subset=['latitude', 'longitude'], inplace=True)
    #convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    earliest_review = df['last_review'].min()
    df['last_review'].fillna(earliest_review, inplace=True)
    # set na values in reviews_per_month to 0
    df['reviews_per_month'].fillna(0, inplace=True)

    return df



"""
load dataframe to transformed table.
"""
def write_to_redshift(df):

    try:
        #convert dataFrame to CSV format and write to Redshift
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False, header=False)

        #load transformed data
        csv_buffer.seek(0)
        cursor.copy_from(csv_buffer, redshift_transformed_table, sep=',', null='')

        conn.commit()
        logger.info(f"Successfully loaded transformed data")

    except Exception as e:
        logger.error(f"Excepction occured while loading transformed data: {e}")
        raise
