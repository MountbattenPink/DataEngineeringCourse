--Raw Table: Stores the original unprocessed data.
-- Transformed Table: Stores the cleaned and transformed data.
CREATE TABLE "public"."raw"(
                               "id" INTEGER NULL,
                               "name" VARCHAR NULL,
                               "host_id" INTEGER NULL,
                               "host_name" VARCHAR NULL,
                               "neighbourhood_group" VARCHAR NULL,
                               "neighbourhood" VARCHAR NULL,
                               "latitude" REAL NULL,
                               "longitude" REAL NULL,
                               "room_type" VARCHAR NULL,
                               "price" INTEGER NULL,
                               "minimum_nights" INTEGER NULL,
                               "number_of_reviews" INTEGER NULL,
                               "last_review" VARCHAR NULL,
                               "reviews_per_month" REAL NULL,
                               "calculated_host_listings_count" INTEGER NULL,
                               "availability_365" INTEGER NULL
) ENCODE AUTO;

CREATE TABLE "public"."transformed"(
                                       "id" INTEGER NOT NULL,
                                       "name" VARCHAR NULL,
                                       "host_id" INTEGER NULL,
                                       "host_name" VARCHAR NULL,
                                       "neighbourhood_group" VARCHAR NULL,
                                       "neighbourhood" VARCHAR NULL,
                                       "latitude" REAL NOT NULL,
                                       "longitude" REAL NOT NULL,
                                       "room_type" VARCHAR NULL,
                                       "price" INTEGER NOT NULL,
                                       "minimum_nights" INTEGER NULL,
                                       "number_of_reviews" INTEGER NULL,
                                       "last_review" TIMESTAMP NOT NULL,
                                       "reviews_per_month" REAL NOT NULL,
                                       "calculated_host_listings_count" INTEGER NULL,
                                       "availability_365" INTEGER NULL
) ENCODE AUTO;





--Implement data quality checks using Redshift SQL to ensure no critical fields like price,
--minimum_nights, or availability_365 contain NULL values.
--• Write SQL queries to validate the data consistency after loading

select COUNT(id) from transformed where minimum_nights is null OR availability_365 is null;