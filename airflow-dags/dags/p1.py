from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="basic_test_dag",
    schedule=None,
    start_date=pendulum.datetime(2025, 8, 15, tz="UTC"),
    catchup=False,
    tags=["basic_test"],
) as dag:
    
    # Tarea 1
    hello_task = BashOperator(
        task_id="print_hello",
        bash_command="echo '¡Hola desde Airflow!'",
    )
    
    # Tarea 2
    date_task = BashOperator(
        task_id="print_date",
        bash_command="date",
    )
    
    # Define el flujo del DAG
    hello_task >> date_task