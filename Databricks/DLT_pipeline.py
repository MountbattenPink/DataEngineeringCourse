import dlt
from pyspark.sql.functions import *
from pyspark.sql.window import Window

# Bronze Table: Store raw, untransformed data in a Delta Lake table.
@dlt.table(
    name="bronze_table",
    comment="Store raw, untransformed data in a Delta Lake table.",
)
def bronze_table():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        .load("dbfs:/mnt/data_engineering/nyc_airbnb_db/raw")
    )

# Silver Table: Store transformed, clean data.
@dlt.table(
    name="silver_table",
    comment="Store transformed, clean data.",
)
@dlt.expect_or_drop("valid_price", "price > 0")
@dlt.expect_or_drop("valid_latitude_longitude", "latitude IS NOT NULL AND longitude IS NOT NULL")
def silver_table():
    bronze_df = dlt.read("bronze_table")
    return (
        bronze_df
        .withColumn("reviews_per_month",
                    when(col("reviews_per_month").isNull(),
                         lit(0))
                    .otherwise(col("reviews_per_month")))
        .withColumn("last_review",
                    when(col("last_review").isNull(),
                         spark_min("last_review").over(Window.orderBy("last_review")))
                    .otherwise(col("last_review")))
    )

@dlt.table(
    name="silver_table_with_constraints",
    comment="constraint for validation separate columns in silver table"
)
@dlt.expect_or_drop("non_null_minimum_nights", "minimum_nights IS NOT NULL")
@dlt.expect_or_drop("non_null_availability_365", "availability_365 IS NOT NULL")
def silver_table_with_constraints():
    return dlt.read("silver_table")
