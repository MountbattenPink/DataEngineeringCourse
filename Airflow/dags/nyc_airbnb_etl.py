from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.utils.session import provide_session
from pyhocon import ConfigFactory
from airflow.models import Param, Connection
from task_callables import ingest_data, log_error, transform, check_quality, log_success_execution, log_error_to_file

config = ConfigFactory.parse_file('../resources/application.conf')
connection_id = 'postgres_connection'

default_args = {
    'owner': 'olsh',
    'retries': 5,
    'retry_delay': timedelta(minutes=2),
    'on_failure_callback': log_error_to_file
}


@provide_session
def get_connection(host, user, password, session=None):
    old_connection = session.query(Connection).filter(Connection.conn_id == connection_id).first()
    if old_connection:
        session.delete(old_connection)
        session.commit()

    new_connection = Connection(
        conn_id=connection_id,
        conn_type='postgres',
        host=host,
        login=user,
        password=password,
        schema='airflow_etl'
    )
    session.add(new_connection)
    session.commit()


with DAG(
    dag_id="nyc_airbnb_etl",
    default_args=default_args,
    start_date=datetime(year=2024, month=1, day=1, hour=0, minute=0),
    schedule="0 0 * * *",
    catchup=True,
    max_active_runs=1,
    render_template_as_native_obj=True,
    params={
            'input_file_path': Param(config.get_string("app.input_file_path"), type='string'),
            'output_file_path': Param(config.get_string("app.output_file_path"), type='string'),
            'host': Param(config.get_string("app.db_connection.host"), type='string'),
            'user': Param(config.get_string("app.db_connection.user"), type='string'),
            'password': Param(config.get_string("app.db_connection.pass"), type='string')
        },

) as dag:
    # Ingest Data from the CSV File:
    # o Use an Airflow PythonOperator to read the AB_NYC_2019.csv file from the raw directory.
    # o Implement basic error handling to ensure the file exists and is readable.
    # o Store the raw data in a Pandas DataFrame for further processing.
    ingest_csv_data = PythonOperator(
        task_id='ingest_data_task',
        python_callable=ingest_data,
        op_kwargs={'csv_path': "{{ params.input_file_path }}"},
        on_failure_callback=log_error,
        provide_context=True,
        dag=dag
    )

    # Transform the Data:
    # o Use a PythonOperator to perform the following transformations:
    # ▪ Filter out rows where price is 0 or negative.
    # ▪ Convert last_review to a datetime object.
    # ▪ Handle missing (if any) last_review dates by filling them with the earliest
    # date in the dataset or a default date.
    # ▪ Handle missing values in reviews_per_month by filling them with 0.
    # ▪ Drop any rows(if any) with missing latitude or longitude values.
    # o Save the transformed DataFrame to a CSV file in the transformed directory.
    transform = PythonOperator(
        task_id='transform_data_task',
        python_callable=transform,
        op_kwargs={
            'output_file_path': "{{ params.output_file_path }}"
        },
        on_failure_callback=log_error,
        provide_context=True,
        dag=dag
    )


    # 6 Load Data into PostgreSQL:
    # o Use the PostgresOperator to load the transformed data from the DataFrame into
    # the airbnb_listings table in PostgreSQL.
    # o Ensure that the data is loaded correctly, handling any potential conflicts or issues
    # (e.g., duplicate entries)

    # FROM DOCUMENTATION:
    # Previously, PostgresOperator was used to perform this kind of operation.
    # But at the moment PostgresOperator is deprecated and will be removed in
    # future versions of the provider. Please consider to switch to SQLExecuteQueryOperator as soon as possible.
    get_db_connection = PythonOperator(
        task_id='get_db_connection_task',
        python_callable=get_connection,
        op_kwargs={
            'host': "{{ params.host }}",
            'user': "{{ params.user }}",
            'password': "{{ params.password }}"
        },
        dag=dag,
    )

    load_to_pg = SQLExecuteQueryOperator(
        task_id='load_to_pg_task',
        conn_id=connection_id,
        sql="sql/nyc_airbnb_inserts.sql",
    )

    # 7 Implement Data Quality Checks:
    # o Use a PythonOperator to perform the following data quality checks:
    # ▪ Ensure the number of records in the PostgreSQL table matches the
    # expected number from the transformed CSV.
    # ▪ Validate that there are no NULL values in the price, minimum_nights, and
    # availability_365 columns.
    # o If any data quality check fails, use the BranchPythonOperator to branch the
    # workflow:
    # ▪ If checks pass, proceed to the next task.
    # ▪ If checks fail, trigger a task that logs the error and stops further processing.
    check_quality = PythonOperator(
        task_id='check_data_quality_task',
        python_callable=check_quality,
            op_kwargs={
                'db_connection_id': connection_id
            },
        provide_context=True,
        dag=dag

    )

    quality_check_branch_task = BranchPythonOperator(
        task_id='quality_check_branch',
        python_callable=lambda: 'success_task',
        provide_context=True
    )

    success_task = PythonOperator(
        task_id='success_task',
        python_callable=log_success_execution
    )

    failure_task = PythonOperator(
        task_id='log_error_and_stop',
        python_callable=log_error
    )

    # 10. Document the Workflow:
    # o Add comments and documentation strings to each task in the DAG.
    # o Create a README file explaining how to run the DAG, configure parameters, and
    # interpret the results.

    # 11. Run and Test the DAG:
    # o Trigger the DAG manually and ensure each task executes as expected.
    # o Monitor the logs to verify that data is correctly ingested, transformed, and loaded.
    # o Validate the contents of the PostgreSQL table against the transformed CSV file.

    # 12. Optimize and Refactor:
    # o Review the DAG for optimization opportunities (e.g., parallel execution of
    # independent tasks, reducing unnecessary I/O operations).
    # o Refactor code to improve readability, maintainability, and performance.


    # Deliverables:
    # 1. The Airflow DAG Script:
    # The Python script that defines the Airflow DAG (nyc_airbnb_etl.py), including:
    # ▪ The DAG definition.
    # ▪ Tasks for ingesting, transforming, and loading the data.
    # ▪ The failure callback function that logs errors to a file.
    # 2. The README file explaining how to run the DAG, configure parameters, and interpret the
    # results.

    ingest_csv_data >> transform >> load_to_pg >> check_quality >> quality_check_branch_task
    quality_check_branch_task >> [success_task, failure_task]

    get_db_connection >> load_to_pg






