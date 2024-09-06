from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, lag, to_date, lit, coalesce, when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, DateType
from pyspark.sql.functions import col
import logging

def get_logger():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s %(message)s')
    formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s')
    logger = logging.getLogger()
    file_handler = logging.FileHandler('etl.log')
    file_handler.setLevel(logging.WARN)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


logger = get_logger()

spark = SparkSession.builder.appName("etl-pipeline-data-engineering-course").master("local").getOrCreate()
input_schema = StructType([StructField('id', StringType(), True),
                     StructField('name', StringType(), True),
                     StructField('host_id', StringType(), True),
                     StructField('host_name', StringType(), True),
                     StructField('neighbourhood_group', StringType(), True),
                     StructField('neighbourhood', StringType(), True),
                     StructField('latitude', StringType(), True),
                     StructField('longitude', StringType(), True),
                     StructField('room_type', StringType(), True),
                     StructField('price', IntegerType(), True),
                     StructField('minimum_nights', IntegerType(), True),
                     StructField('number_of_reviews', IntegerType(), True),
                     StructField('last_review', StringType(), True),
                     StructField('reviews_per_month', IntegerType(), True),
                     StructField('calculated_host_listings_count', StringType(), True),
                     StructField('availability_365', StringType(), True),
                     StructField('last_review_date', DateType(), True),
                     StructField('price_category', StringType(), True),
                     StructField('price_per_review', DoubleType(), True)])

processed_df = spark.read.option("header", "true").schema(input_schema).parquet("processed").withColumn("file_name", input_file_name())

archived_files = set()
archived_files_log_path = 'resources/archived_files.log'
# creates initial set of files which were processed during previous runs
# and should be dismissed this time
def get_archived_files_names():
    try:
        with open(archived_files_log_path, 'r') as file:
            for line in file:
                archived_files.add(line.strip())
    except Exception as e:
        logger.error("Error reading from archived files log")
        raise e



get_archived_files_names()

try:
    ingest_csv = ((spark
                   .readStream.format("csv")
                   .schema(input_schema)
                   .option("header", "true")
                   .load("raw"))
                  .withColumn("file_name", input_file_name()))

except Exception as e:
    logger.error("Error reading from raw directory")
    raise e

#3. Transform the Data using PySpark:
# filtering, transforming for batch and only after - merging with existing data
# only if file is not already archived
def merge_with_processed_data(batch_df, _):
    global processed_df, archived_files
    file_name = batch_df.head()["file_name"]
    logger.info(f"Handling file: {file_name}")
    if (file_name not in archived_files):
        filtered_df = (batch_df
                       .filter(batch_df["price"] > 0)
                       .withColumn('last_review_date', coalesce(to_date(col("last_review")), to_date(lit('1970-01-01')))))
        cleaned = (filtered_df.fillna({"reviews_per_month": 0})
                   .fillna({"availability_365": 0})
                   .dropna(subset=["latitude", "longitude", "availability_365"]))
        transformed = (cleaned
                       .withColumn("price_category",
                                   when(col("price") <= 10, lit("budget"))
                                   .when(col("price") >= 100, lit("luxury"))
                                   .otherwise(lit("mid-range")))
                     .withColumn("price_per_review",
                                   when(col("number_of_reviews") > 0, col("price") / col("number_of_reviews"))
                                   .otherwise(lit(None)))
                       )
        processed_df = processed_df.union(transformed).dropDuplicates(["host_id", "latitude", "longitude", "room_type", "price"])
        archived_files.add(file_name)
        logger.info(f"Successfully handled file {file_name}")


try:
    new_data_df = (ingest_csv.writeStream
                   .trigger(processingTime="10 seconds")
                   .foreachBatch(merge_with_processed_data)
                   .outputMode("append").start())

    new_data_df.processAllAvailable()

except Exception as e:
    logger.error("Error handling streamed dataframe")
    raise e




try:
    processed_df.createOrReplaceTempView("merged_data")
    listing_by_neighbourhood_group = spark.sql("""
        SELECT neighbourhood_group, COUNT(id) AS listing_count
        FROM merged_data
        GROUP BY neighbourhood_group
        ORDER BY listing_count DESC
    """)
    top10_most_expensive_listings = spark.sql("""
        SELECT name, host_id, host_name, neighbourhood_group, neighbourhood, price AS listing
        FROM merged_data
        ORDER BY price DESC LIMIT 10
    """)
    avg_price_per_room_type = spark.sql("""
        SELECT neighbourhood_group, room_type, AVG(price) AS avg_price
        FROM merged_data
        GROUP BY neighbourhood_group, room_type
    """)

    #Repartition the Data
    repartitioned_df = processed_df.repartition("neighbourhood_group")
    #Save the Data using PySpark
    repartitioned_df.write.parquet("processed", mode="overwrite")
    spark.sql("REFRESH TABLE merged_data")

    #sql queries
    listing_by_neighbourhood_group.repartition(1).write.mode("overwrite").parquet("sql_queries/listing_by_neighbourhood_group")
    top10_most_expensive_listings.repartition(1).write.mode("overwrite").parquet("sql_queries/top10_most_expensive_listings")
    avg_price_per_room_type.repartition(1).write.mode("overwrite").parquet("sql_queries/avg_price_per_room_type")

    with open(archived_files_log_path, 'w') as file:
        for item in archived_files:
            file.write(f"{item}\n")

except Exception as e:
    logger.error("Error handling querying and storing data")
    raise e


#Data Quality Checks using PySpark:
actual_repartitioned_row_count = repartitioned_df.count()
expected_repartitioned_row_count = 48873
# Row Count Validation
assert actual_repartitioned_row_count == expected_repartitioned_row_count, f"Resulted dataframe count should be {expected_repartitioned_row_count} but was {actual_repartitioned_row_count}"

assert repartitioned_df.filter(repartitioned_df["price"].isNull()).count() == 0, "Empty price values found"

assert repartitioned_df.filter(repartitioned_df["minimum_nights"].isNull()).count() == 0, "Empty minimum_nights values found"

assert repartitioned_df.filter(repartitioned_df["availability_365"].isNull()).count() == 0, "Empty availability_365 values found"

