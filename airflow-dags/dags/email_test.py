from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

# Configuración por defecto
default_args = {
    "owner": "data-team",
    "depends_on_past": False,
    "email": ["jndrs2111@gmail.com"],  # destinatario
    "email_on_failure": True,          # ✅ se envía correo si falla
    "email_on_retry": False,
    "retries": 1,                      # número de reintentos
    "retry_delay": timedelta(minutes=1),
}

# Funciones ETL
def extract():
    print("📥 Extrayendo datos...")

def transform():
    print("🛠️ Transformando datos...")

def load():
    print("📦 Cargando datos al destino final...")

def simular_fallo():
    print("💥 Esta tarea va a fallar intencionalmente...")
    raise ValueError("Fallo forzado para probar retries ⚡")

# DAG
with DAG(
    dag_id="etl_template_con_fallo",
    description="Plantilla ETL con simulación de fallo y reintentos",
    default_args=default_args,
    start_date=datetime(2025, 8, 15),
    schedule=None,
    catchup=False,
    tags=["etl", "template", "test-fallo"],
) as dag:

    inicio = EmptyOperator(task_id="inicio")

    extract_task = PythonOperator(
        task_id="extract",
        python_callable=extract,
    )

    transform_task = PythonOperator(
        task_id="transform",
        python_callable=transform,
    )

    fallo_task = PythonOperator(
        task_id="simular_fallo",
        python_callable=simular_fallo,
    )

    load_task = PythonOperator(
        task_id="load",
        python_callable=load,
    )

    fin = EmptyOperator(task_id="fin")

    # Flujo del DAG
    inicio >> extract_task >> transform_task >> fallo_task >> load_task >> fin