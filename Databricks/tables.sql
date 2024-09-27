CREATE TABLE bronze_table
(id LONG,
 host_name STRING,
 latitude DOUBLE,
 price LONG,
 host_id LONG,
 last_review STRING,
 availability_365 LONG,
 neighbourhood STRING,
 room_type STRING,
 number_of_reviews LONG,
 calculated_host_listings_count LONG,
 name STRING,
 neighbourhood_group STRING,
 longitude DOUBLE,
 minimum_nights LONG,
 reviews_per_month DOUBLE)
    USING delta
LOCATION '/mnt/data_engineering_course/nyc_airbnb_db/bronze_table';


CREATE TABLE silver_table
(id LONG,
 host_name STRING,
 latitude DOUBLE,
 price LONG,
 host_id LONG,
 last_review DATE,
 availability_365 LONG,
 neighbourhood STRING,
 room_type STRING,
 number_of_reviews LONG,
 calculated_host_listings_count LONG,
 name STRING,
 neighbourhood_group STRING,
 longitude DOUBLE,
 minimum_nights LONG,
 reviews_per_month DOUBLE)
    USING delta
LOCATION '/mnt/data_engineering_course/nyc_airbnb_db/silver_table';