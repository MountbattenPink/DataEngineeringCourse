%pip install delta-spark

import logging
from pyspark.sql.types import (
    StructType, StructField, IntegerType, StringType, DoubleType, DateType
)
from pyspark.sql.functions import col, when, lit, min as spark_min
from pyspark.sql.window import Window

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Step 1: Set Up the Environment
    # List available catalogs to ensure the needed catalog exists
    display(spark.sql("SHOW CATALOGS"))

    # Step 2: Create a Databricks Database
    # create database (schema) inside catalog and use it
    spark.sql("USE CATALOG data_engiineering_course")
    spark.sql("CREATE SCHEMA IF NOT EXISTS nyc_airbnb_db")
    spark.sql("USE nyc_airbnb_db")


    # Define the schema based on CSV New York City Airbnb Open Data from Kaggle
    # it's needed for consistency in tables during ingesting files
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("host_id", IntegerType(), True),
        StructField("host_name", StringType(), True),
        StructField("neighbourhood_group", StringType(), True),
        StructField("neighbourhood", StringType(), True),
        StructField("latitude", DoubleType(), True),
        StructField("longitude", DoubleType(), True),
        StructField("room_type", StringType(), True),
        StructField("price", IntegerType(), True),
        StructField("minimum_nights", IntegerType(), True),
        StructField("number_of_reviews", IntegerType(), True),
        StructField("last_review", DateType(), True),
        StructField("reviews_per_month", DoubleType(), True),
        StructField("calculated_host_listings_count", IntegerType(), True),
        StructField("availability_365", IntegerType(), True)
    ])


    # paths for source, destination data and checkpoints
    source_path = "dbfs:/mnt/data_engineering_course/nyc_airbnb_db/raw"

    bronze_table_path = "dbfs:/mnt/data_engineering_course/nyc_airbnb_db/bronze_table"
    bronze_checkpoint_path = "dbfs:/mnt/data_engineering_course/nyc_airbnb_db/checkpoints/bronze_table_checkpoint"

    silver_table_path = "dbfs:/mnt/data_engineering_course/nyc_airbnb_db/silver_table"
    silver_checkpoint_path = "dbfs:/mnt/data_engineering_course/nyc_airbnb_db/checkpoints/silver_table_checkpoint"


    # Step 3: Data Ingestion Using Auto Loader
    # Read the data using Auto Loader
    df = (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .option("cloudFiles.schemaLocation", bronze_checkpoint_path)
        .schema(schema)
        .load(source_path)
    )

    # Write the data to a Bronze Delta Table
    bronze_stream = (
        df.writeStream
        .format("delta")
        .outputMode("complete")
        .option("checkpointLocation", bronze_checkpoint_path)
        .start(bronze_table_path)
    )

    # Process all available data and then proceed
    bronze_stream.processAllAvailable()
    logger.info("BRONZE TABLE PROCESSED")

    # 5. Data Transformation
    # Read the Bronze table
    bronze_df = spark.read.format("delta").load(bronze_table_path)

    # (order of operations is changed to minimize data size first and so optimize )
    # Filter out rows where price is 0 or negative.
    # Convert last_review to a valid date format and fill missing values with the earliest available date.
    # Handle missing values in reviews_per_month by setting them to 0.
    # Drop rows with missing latitude or longitude
    silver_df = bronze_df.filter(col("price") > 0) \
        .dropna(subset=["latitude", "longitude"]) \
        .withColumn("reviews_per_month", when(col("reviews_per_month").isNull(), lit(0)).otherwise(col("reviews_per_month"))) \
        .withColumn("last_review", when(col("last_review").isNull(), spark_min("last_review").over(Window.orderBy("last_review"))).otherwise(col("last_review")))

    logger.info("SILVER TRANSFORMATIONS APPLIED")

    # Step 6: Store the transformed data in a Silver Delta Table
    silver_df.write.format("delta").mode("overwrite").save(silver_table_path)
    logger.info("SILVER TABLE WRITTEN")

    # Step 7: Data Quality Checks:
    # Implement data quality checks using Delta Lake’s constraint validation and Databricks
    # SQL to ensure no critical fields like price, minimum_nights, or availability_365 contain
    # NULL values.
    spark.sql(f"""ALTER TABLE delta.`{silver_table_path}`
    ADD CONSTRAINT price_non_null CHECK (price IS NOT NULL)""")

    spark.sql(f"""ALTER TABLE delta.`{silver_table_path}`
    ADD CONSTRAINT minimum_nights_non_null CHECK (minimum_nights IS NOT NULL)""")

    spark.sql(f"""ALTER TABLE delta.`{silver_table_path}`
    ADD CONSTRAINT availability_365_non_null CHECK (availability_365 IS NOT NULL)""")

    logger.info("SILVER TABLE PROCESSED")


    # 8. Implement Streaming and Monitoring:
    # Use Structured Streaming to monitor data changes in real-time. Set up the streaming logic
    # to listen to changes in the bronze table and propagate those changes to the silver table.
    # Monitor the data stream and transformations using Databricks Jobs for scheduling and
    # execution.
    # 9. Automation Using Databricks Jobs:
    # Set up Databricks Jobs to automate the daily ingestion and transformation process.
    # Use Delta Live Tables (DLT) for simplifying the management of production streaming jobs.
    # Create a schedule for your job using Databricks Job Scheduler.
    def send_bronze_to_silver():
        bronze_streaming_df = spark.readStream.format("delta").load(bronze_table_path)

        silver_streaming_df = (bronze_streaming_df \
                               .filter(col("price") > 0) \
                               .withColumn("last_review",  \
                                                when( \
                                                    col("last_review").isNull(), \
                                                    spark_min("last_review").over(Window.orderBy("last_review")) \
                                                ).otherwise(col("last_review"))) \
                               .withColumn("reviews_per_month", \
                                                when( \
                                                    col("reviews_per_month").isNull(), \
                                                    lit(0) \
                                                ).otherwise(col("reviews_per_month"))) \
                               .dropna(subset=["latitude", "longitude"]))

        silver_stream = (
            silver_streaming_df.writeStream
            .format("delta")
            .outputMode("append")
            .option("checkpointLocation", silver_checkpoint_path)
            .start(silver_table_path)
        )

        return silver_stream

    silver_stream = send_bronze_to_silver()

# 10. Error Handling and Monitoring:
# Implement error handling mechanisms using Try-Catch in PySpark or custom logging mechanisms.
except Exception as e:
    logger.error(f"Exception: {e.__str__()}")

# Set up Delta Lake’s Time Travel to recover from accidental changes or errors by querying
# previous versions of data:

# previous_timestamp_df = spark.read.format("delta") \
#    .option("timestampAsOf", "2024-09-25T00:00:00.000Z") \
#    .load("/mnt/data_engineering_course/nyc_airbnb_db/silver_table")
# display(previous_timestamp_df)