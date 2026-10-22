import os
import io
from pathlib import Path

import boto3
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# 1. EXTRACT: leer los datos originales
BASE_DIR = Path(__file__).resolve().parent
archivo_csv = BASE_DIR / "data" / "clientes.csv"

df = pd.read_csv(archivo_csv)
print("Datos originales:")
print(df)


# 2. TRANSFORM: categorizar edades y normalizar compras
df["categoria_edad"] = pd.cut(
    df["edad"],
    bins=[0, 25, 50, 100],
    labels=["Joven", "Adulto", "Senior"],
    include_lowest=True
)

scaler = MinMaxScaler()
df["score_promocional"] = (
    scaler.fit_transform(df[["monto_compras"]]).flatten() * 100
).round(2)

print("\nDatos transformados:")
print(df)


# 3. LOAD: guardar en formato Parquet
salida = BASE_DIR / "processed" / "clientes_procesados.parquet"
salida.parent.mkdir(parents=True, exist_ok=True)

buffer = io.BytesIO()
df.to_parquet(buffer, index=False, engine="pyarrow")
buffer.seek(0)

with open(salida, "wb") as archivo:
    archivo.write(buffer.getvalue())

print(f"\nArchivo Parquet guardado en: {salida}")


# 4. Carga opcional a AWS S3
bucket = os.getenv("AWS_BUCKET")
region = os.getenv("AWS_REGION", "us-east-1")
clave_s3 = "processed/clientes_procesados.parquet"

if bucket:
    s3 = boto3.client("s3", region_name=region)
    buffer.seek(0)
    s3.upload_fileobj(buffer, bucket, clave_s3)
    print(f"Archivo subido a s3://{bucket}/{clave_s3}")
else:
    print("AWS_BUCKET no está configurado; se guardó solo localmente.")
