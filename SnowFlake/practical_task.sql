USE ROLE accountadmin;

-- creating warehouse for computing
CREATE OR REPLACE WAREHOUSE nyc_warehouse WITH 
WAREHOUSE_SIZE='XSMALL'
AUTO_SUSPEND=20
AUTO_RESUME=TRUE;

USE WAREHOUSE nyc_warehouse;

-- creating DB for storing all objects under schema public
CREATE OR REPLACE DATABASE nyc_airbnb;
USE DATABASE nyc_airbnb;
USE SCHEMA nyc_airbnb.public;

-- table for ingested data (before cleaning and transforming)
create or replace TABLE nyc_airbnb.public.airbnb_raw (
	ID VARCHAR,
	NAME VARCHAR,
	HOST_ID VARCHAR,
	HOST_NAME VARCHAR,
	NEIGHBOURHOOD_GROUP VARCHAR,
	NEIGHBOURHOOD VARCHAR,
	LATITUDE FLOAT,
	LONGITUDE FLOAT,
	ROOM_TYPE VARCHAR,
	PRICE NUMBER,
	MINIMUM_NIGHTS NUMBER,
	NUMBER_OF_REVIEWS NUMBER,
	LAST_REVIEW VARCHAR,
	REVIEWS_PER_MONTH FLOAT,
	CALCULATED_HOST_LISTINGS_COUNT NUMBER,
	AVAILABILITY_365 NUMBER
) DATA_RETENTION_TIME_IN_DAYS=1;


CREATE OR REPLACE FILE FORMAT nyc_file_format
    TYPE = 'CSV';


--setting up communication with s3 bucket updates
CREATE STORAGE INTEGRATION nyc_integration
  TYPE = EXTERNAL_STAGE
  STORAGE_PROVIDER = 'S3'
  ENABLED = TRUE
  STORAGE_AWS_ROLE_ARN = 'arn:aws:iam::700128968765:role/forsnowflake'
  STORAGE_ALLOWED_LOCATIONS = ('s3://data-engineering-course-olsh/');

DESC INTEGRATION nyc_integration;

GRANT CREATE STAGE ON SCHEMA public TO ROLE accountadmin;
GRANT USAGE ON INTEGRATION nyc_integration TO ROLE accountadmin;

CREATE OR REPLACE STAGE nyc_stage
  STORAGE_INTEGRATION = nyc_integration
  URL = 's3://data-engineering-course-olsh/'
  FILE_FORMAT = nyc_file_format;


-- pipeline for reading newly added files to s3 bucket
CREATE OR REPLACE PIPE nyc_airbnb.public.nyc_pipe
  AUTO_INGEST = TRUE
  AS
    COPY INTO nyc_airbnb.public.airbnb_raw
      FROM @nyc_stage
      FILE_FORMAT = nyc_file_format;

ALTER PIPE nyc_airbnb.public.nyc_pipe REFRESH;

create or replace stream updatescheck on table airbnb_raw;


-- table for storing already transformed data
-- has the same structure as airbnb_raw except for LAST_REVIEW is of type DATE
create or replace TABLE nyc_airbnb.public.airbnb_transformed (
	ID VARCHAR,
	NAME VARCHAR,
	HOST_ID VARCHAR,
	HOST_NAME VARCHAR,
	NEIGHBOURHOOD_GROUP VARCHAR,
	NEIGHBOURHOOD VARCHAR,
	LATITUDE FLOAT,
	LONGITUDE FLOAT,
	ROOM_TYPE VARCHAR,
	PRICE NUMBER,
	MINIMUM_NIGHTS NUMBER,
	NUMBER_OF_REVIEWS NUMBER,
	LAST_REVIEW DATE,
	REVIEWS_PER_MONTH FLOAT,
	CALCULATED_HOST_LISTINGS_COUNT NUMBER,
	AVAILABILITY_365 NUMBER
)  DATA_RETENTION_TIME_IN_DAYS=1;


-- transforms data and moves from airbnb_raw table to airbnb_transformed
--scheduled to run at midnights UTC
CREATE OR REPLACE TASK nyc_task
    WAREHOUSE = nyc_warehouse
    SCHEDULE = 'USING CRON 0 0 * * * UTC'
AS
BEGIN
        -- Filter out rows where price is 0 or negative
        -- Drop rows with missing latitude or longitude
        delete from airbnb_raw
        where price<=0 OR latitude is null OR longitude is null;

        -- Handle missing values in reviews_per_month by setting them to 0
        update airbnb_raw
        set reviews_per_month=0 where reviews_per_month is null;

        -- Convert the last_review to a valid date format:
        -- steps:
        -- 1. adding column 'last_review_date' of type DATE
        -- 2. trying to set 'last_review_date' with value of converted 'last_review' column or default
        -- 3. deleting column 'last_review' and renaming 'last_review_date' to 'last_review'
        alter table airbnb_raw add column last_review_date DATE;
        update airbnb_raw
        set last_review_date = COALESCE(TO_DATE(last_review), TO_DATE('1970-01-01'));
        alter table airbnb_raw drop column last_review;
        alter table airbnb_raw rename column last_review_date to last_review;

        -- moving all data from raw to transformed table
        insert into airbnb_transformed (ID, NAME, HOST_ID, HOST_NAME, NEIGHBOURHOOD_GROUP, NEIGHBOURHOOD, LATITUDE, LONGITUDE, ROOM_TYPE, PRICE,
                                        MINIMUM_NIGHTS, NUMBER_OF_REVIEWS, LAST_REVIEW, REVIEWS_PER_MONTH, CALCULATED_HOST_LISTINGS_COUNT, AVAILABILITY_365)
        select ID, NAME, HOST_ID, HOST_NAME, NEIGHBOURHOOD_GROUP, NEIGHBOURHOOD, LATITUDE, LONGITUDE, ROOM_TYPE, PRICE,
               MINIMUM_NIGHTS, NUMBER_OF_REVIEWS, LAST_REVIEW, REVIEWS_PER_MONTH, CALCULATED_HOST_LISTINGS_COUNT, AVAILABILITY_365
        from airbnb_raw;
        delete from airbnb_raw;
END;

ALTER TASK nyc_task RESUME;


-- validation for transformed data
CREATE OR REPLACE STREAM nyc_stream_transformed ON TABLE airbnb_transformed;

-- table for logging each task execution status (success/failed) for transformed data validation
CREATE OR REPLACE TABLE logs (
    timstamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status STRING
);

CREATE OR REPLACE TASK log_errors_task
    WAREHOUSE = nyc_warehouse
    SCHEDULE = 'USING CRON 0 0 * * * UTC'
AS
begin
    INSERT INTO logs (status)
    SELECT CASE WHEN errors = 0 THEN 'Success' ELSE 'Validation failed' END AS status
    FROM
        (
            SELECT COUNT(*) as errors
            FROM airbnb_transformed
            WHERE price IS NULL OR minimum_nights IS NULL OR availability_365 IS NULL
        );
    end;


ALTER TASK log_errors_task RESUME;


-- if it's needed to use time travel, run this command:
-- SELECT * 
-- FROM airbnb_transformed 
-- AT (TIMESTAMP => '2024-09-12 HH00:00:00');
        