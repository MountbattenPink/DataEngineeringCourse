-- Task 1: Identify Authors with the Most Published Books
-- Create a CTE that calculates the total number of books each author has published. Then, create
-- SELECT query that will use this CTE, and retrieve a list of authors who have published more than 3
-- books, including the number of books they have published.


with authors_to_books_counts as (select books.author_id, COUNT(books.book_id) as book_count
                                 from books
                                 group by books.author_id)
select authors_to_books_counts.author_id, authors.name, authors_to_books_counts.book_count as books_published
from authors_to_books_counts inner join authors using (author_id)
where authors_to_books_counts.book_count > 3;


-- Task 2: Identify Books with Titles Containing 'The' Using Regular
-- Expressions
-- Create a CTE that identifies books whose titles contain the word "The" in any letter case using
-- regular expressions. For example, books with titles like "The Great Gatsby," "The Shining", "The Old
-- Man and the Sea", or "To the Lighthouse" will meet this criterion. Then, create the SELECT that will
-- this CTE, and retrieve the book titles, corresponding authors, genre, and publication date.

with books_with_the_in_name as (select books.book_id, books.title, books.published_date, books.author_id, books.genre_id
                                from books
                                where books.title ~ '.*the.*')
select books_with_the_in_name.title, books_with_the_in_name.published_date, authors.name, genres.genre_name
from books_with_the_in_name
         inner join authors using (author_id)
         inner join genres using (genre_id);



-- Task 3: Rank Books by Price within Each Genre Using the RANK() Window
-- Function
-- Create a query that ranks books by their price within each genre using the RANK() window
-- function. The goal is to assign a rank to each book based on its price, with the highest-priced book
-- in each genre receiving a rank of 1.

select books.title,
       genres.genre_name,
       books.price,
       RANK() over (partition by books.genre_id order by books.price desc) as rank_by_price
from books
         inner join public.genres using (genre_id);


-- Task 4: Bulk Update Book Prices by Genre
-- Create a stored procedure that increases the prices of all books in a specific genre by a specified
-- percentage. The procedure should also output the number of books that were updated. Use RAISE
-- for it.
-- Potential procedure name: sp_bulk_update_book_prices_by_genre
-- Parameters for procedure:
-- p_genre_id INTEGER
-- p_percentage_change NUMERIC(5, 2)
-- Example of usage:
-- -- This increases the prices of all books in genre ID 3 by 5% and output the updated number
-- CALL sp_bulk_update_book_prices_by_genre(3, 5.00);


create
or replace procedure sp_bulk_update_book_prices_by_genre(p_genre_id INTEGER, p_percentage_change NUMERIC(5, 2))
    language plpgsql
as
$$
begin
    update books
    set price = price + (price * p_percentage_change * 0.01)
    where books.genre_id = p_genre_id;
end;
$$;

CALL sp_bulk_update_book_prices_by_genre(3, 5.00);


-- Task 5: Update Customer Join Date Based on First Purchase
-- Create a stored procedure that updates the join_date of each customer to the date of their first
-- purchase if it is earlier than the current join_date. This ensures that the join_date reflects the true
-- start of the customer relationship.
-- Potential procedure name: sp_update_customer_join_date
-- Parameters for procedure: NONE
-- Example of usage:
-- -- This updates the join dates of customers to reflect the date of their first purchase if earlier than
-- -- the current join date.
-- CALL sp_update_customer_join_date();


create or replace procedure sp_update_customer_join_date()
       language plpgsql
as
$$
begin
    with customers_need_to_update as (
        select sales.customer_id, sales.sale_date
        from sales join public.customers
                    on sales.customer_id = customers.customer_id and sales.sale_date < customers.join_date)

    update customers
    set join_date = customers_need_to_update.sale_date
        from customers_need_to_update
    where customers.customer_id in (select customer_id from customers_need_to_update);
end;
$$;

-- to test it, we can run the following query before and after procedure execution::
select customers.customer_id
from customers join sales on customers.customer_id = sales.customer_id
where customers.join_date > sales.sale_date;
-- if should be non-empty before and empty after
CALL sp_update_customer_join_date();


-- Task 6: Calculate Average Book Price by Genre
-- Create a function that calculates the average price of books within a specific genre.
-- Potential function name: fn_avg_price_by_genre
-- Parameters: p_genre_id INTEGER
-- Return Type: NUMERIC(10, 2)
-- Example of usage:
-- -- This would return the average price of books in the genre with ID 1.
-- SELECT fn_avg_price_by_genre(1);

create or replace function fn_avg_price_by_genre(p_genre_id INTEGER) returns NUMERIC(10, 2)
    language plpgsql
as $$
declare
    avg_price numeric(10, 2);
begin
    select avg(books.price)
    into avg_price
    from books
    where books.genre_id = p_genre_id;
    return avg_price;
end;
$$;

SELECT fn_avg_price_by_genre(1);

-- Task 7: Get Top N Best-Selling Books by Genre
-- Create a function that returns the top N best-selling books in a specific genre, based on total sales
-- revenue.
-- Potential function name: fn_get_top_n_books_by_genre
-- Parameters:
-- p_genre_id INTEGER
-- p_top_n INTEGER
-- Example of usage:
-- This would return the top 5 best-selling books in genre with ID 1
-- SELECT * FROM fn_get_top_n_books_by_genre(1, 5);

create or replace function fn_get_top_n_books_by_genre(p_genre_id INTEGER, p_top_n INTEGER)
    returns table(id integer)
    language plpgsql
as
$$
begin
return query
    with sales_and_prices as (
            select books.book_id, books.price * sales.quantity as current_revenue
            from books join sales on books.genre_id=p_genre_id and books.book_id = sales.book_id
        ),
        books_and_revenues as (
            select sales_and_prices.book_id, sum(sales_and_prices.current_revenue) as total_revenue
            from sales_and_prices
            group by sales_and_prices.book_id
        )
    select books_and_revenues.book_id
    from books_and_revenues
    order by books_and_revenues.total_revenue desc
    limit p_top_n;
end;
$$;

SELECT * FROM fn_get_top_n_books_by_genre(1, 5);


-- Task 8: Log Changes to Sensitive Data
-- Create a trigger that logs any changes made to sensitive data in a Customers table. Sensitive data:
-- first name, last name, email address. The trigger should insert a record into an audit log table each
-- time a change is made. You need to create this log table by yourself.
-- Log table could have such structure:
-- CREATE TABLE CustomersLog (
-- log_id SERIAL PRIMARY KEY,
-- column_name VARCHAR(50),
-- old_value TEXT,
-- new_value TEXT,
-- changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
-- changed_by VARCHAR(50) -- This assumes you can track the user making the change
-- );
-- Potential trigger name: tr_log_sensitive_data_changes
-- Trigger Timing: AFTER UPDATE
-- Trigger Event: ON table Customers

create or replace function log_trigger_function() returns trigger
    language plpgsql
    as $$
        begin
        if new.first_name is distinct from old.first_name then
            insert into customerslog(column_name, old_value, new_value, changed_by)
            values ('first_name',
                    old.first_name,
                    new.first_name,
                    current_user);
        end if;
        if new.last_name is distinct from old.last_name then
            insert into customerslog(column_name, old_value, new_value, changed_by)
            values ('last_name',
                    old.last_name,
                    new.last_name,
                    current_user);
        end if;
        if new.email is distinct from old.email then
            insert into customerslog(column_name, old_value, new_value, changed_by)
            values ('email',
                    old.email,
                    new.email,
                    current_user);
        end if;
    return new;
        end;
    $$;

    create or replace trigger tr_log_sensitive_data_changes
    after update
    on customers
    for each row
    execute procedure log_trigger_function();

-- Task 9: Automatically Adjust Book Prices Based on Sales Volume
-- Create a trigger that automatically increases the price of a book by 10% if the total quantity sold
-- reaches a certain threshold (e.g., 10 units). This helps to dynamically adjust pricing based on the
-- popularity of the book.
-- Potential trigger name: tr_adjust_book_price
-- Trigger Timing: AFTER INSERT
-- Trigger Event: ON table Sales

create or replace function tr_adjust_book_price_trigger_function() returns trigger
    language plpgsql
as $$
    declare threshold_units integer := 10;
begin
    update books
        set price = price * 1.10
    where
        book_id= new.book_id
    and
        --we need namely perform mod operation, otherwise after 10 sales, the book will become more expensive
        --after EVERY sale (this will make polynomial rise).
        --I think, better to rise the price after each 10 sales.
        (select count(sale_id) from sales where sales.book_id=new.book_id) % threshold_units = 0;
return new;
end;
$$;

create or replace trigger tr_adjust_book_price
    after insert
    on sales
    for each row
execute procedure tr_adjust_book_price_trigger_function();


-- Task 10: Archive Old Sales Records
-- Create a stored procedure that uses a cursor to iterate over sales records older than a specific
-- date, move them to an archive table (SalesArchive), and then delete them from the original
-- Sales table.
-- Potential procedure name: sp_archive_old_sales
-- Parameters: p_cutoff_date DATE
-- Potential Steps:
-- 1. Declare a cursor to fetch sales records older than a given date.
-- 2. For each record, insert it into the SalesArchive table.
-- 3. After inserting record to SalesArchive table, delete the record from the Sales table.
-- The procedure should take the cutoff date as an input parameter.
-- For this task, you will need to create a table called SalesArchive. The structure of this table
-- will be the same as the Sales table.

