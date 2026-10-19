import boto3
import pandas as pd
import json
import io

# 1. Configurar cliente Boto3
s3_client = boto3.client('s3', region_name='us-east-1')

# ⚠️ Sustituye con el nombre exacto de tu bucket en AWS Academy
BUCKET_NAME = 'aws-academy-datacenter-bucket-23-02-015' 

def cargar_y_perfilar_s3(file_key, is_json=False):
    """Descarga archivos desde S3 directamente a memoria y perfila los datos."""
    print(f"\n==================================================")
    print(f"=== Procesando objeto S3: {file_key} ===")
    print(f"==================================================")
    
    try:
        # Lectura directa desde S3 en memoria sin guardar en disco
        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_key)
        body = response['Body'].read()
        
        if is_json:
            data_json = json.loads(body.decode('utf-8'))
            df = pd.json_normalize(data_json)
        else:
            df = pd.read_csv(io.BytesIO(body))
            
        print(f"\n📌 Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
        print("\n📌 Tipos de Datos (df.dtypes):")
        print(df.dtypes)
        print("\n📌 Estadísticos Descriptivos (df.describe):")
        print(df.describe().T)
        
        return df
    except Exception as e:
        print(f"❌ Error al procesar el archivo {file_key}: {e}")
        return None

if __name__ == "__main__":
    # Ejecución de ingesta para CSV y JSON
    df_transacciones = cargar_y_perfilar_s3('raw/transacciones.csv')
    df_logs = cargar_y_perfilar_s3('raw/logs_m2m.json', is_json=True)
