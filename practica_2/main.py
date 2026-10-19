import pandas as pd
import numpy as np

# ==========================================
# 0. SIMULACIÓN DE INGESTA OPERATIVA (OLTP)
# ==========================================
np.random.seed(101)
n = 500

df_operativo = pd.DataFrame({
    'id_transaccion': range(1000, 1000 + n),
    'fecha_hora': pd.date_range(start='2026-01-01', periods=n, freq='h'),
    'cliente_nombre': np.random.choice(['Juan Pérez', 'María Gómez', 'Carlos Ruiz'], n),
    'cliente_ciudad': np.random.choice(['Puebla', 'CDMX', 'Guadalajara'], n),
    'producto_nombre': np.random.choice(['Laptop Dell', 'Mouse Logi', 'Monitor LG'], n),
    'categoria_prod': np.random.choice(['Cómputo', 'Accesorios', 'Pantallas'], n),
    'precio_unitario': np.random.choice([15000.0, 350.0, 4500.0], n),
    'cantidad': np.random.randint(1, 5, n)
})

print("=== 1. DATOS OPERATIVOS (OLTP - Muestra) ===")
print(df_operativo.head())
print("\n" + "="*50 + "\n")

# ==========================================
# 1. DIMENSIÓN CLIENTE (Dim_Cliente)
# ==========================================
dim_cliente = df_operativo[['cliente_nombre', 'cliente_ciudad']].drop_duplicates().reset_index(drop=True)
dim_cliente['id_cliente_sk'] = dim_cliente.index + 1  # Surrogate Key

print("=== DIMENSIÓN CLIENTE (Dim_Cliente) ===")
print(dim_cliente)
print("\n" + "="*50 + "\n")

# ==========================================
# 2. DIMENSIÓN PRODUCTO (Dim_Producto)
# ==========================================
dim_producto = df_operativo[['producto_nombre', 'categoria_prod', 'precio_unitario']].drop_duplicates().reset_index(drop=True)
dim_producto['id_producto_sk'] = dim_producto.index + 1  # Surrogate Key

print("=== DIMENSIÓN PRODUCTO (Dim_Producto) ===")
print(dim_producto)
print("\n" + "="*50 + "\n")

# ==========================================
# 3. DIMENSIÓN TIEMPO (Dim_Tiempo) [EXTENDIDA]
# ==========================================
dim_tiempo = pd.DataFrame({'fecha_completa': df_operativo['fecha_hora'].dt.date.unique()})
dim_tiempo['fecha_completa'] = pd.to_datetime(dim_tiempo['fecha_completa'])
dim_tiempo['id_tiempo_sk'] = dim_tiempo['fecha_completa'].dt.strftime('%Y%m%d').astype(int)
dim_tiempo['anio'] = dim_tiempo['fecha_completa'].dt.year
dim_tiempo['mes'] = dim_tiempo['fecha_completa'].dt.month
dim_tiempo['dia'] = dim_tiempo['fecha_completa'].dt.day

# Atributos extendidos (Requisito de la práctica)
dim_tiempo['nombre_dia'] = dim_tiempo['fecha_completa'].dt.day_name()
dim_tiempo['es_fin_semana'] = dim_tiempo['fecha_completa'].dt.dayofweek.isin([5, 6])

print("=== DIMENSIÓN TIEMPO (Dim_Tiempo - Muestra) ===")
print(dim_tiempo.head())
print("\n" + "="*50 + "\n")

# ==========================================
# 4. TABLA DE HECHOS (Fact_Ventas) [EXTENDIDA]
# ==========================================
fact_ventas = df_operativo.merge(dim_cliente, on=['cliente_nombre', 'cliente_ciudad'])\
    .merge(dim_producto, on=['producto_nombre', 'categoria_prod', 'precio_unitario'])

fact_ventas['fecha_completa'] = pd.to_datetime(fact_ventas['fecha_hora'].dt.date)
fact_ventas = fact_ventas.merge(dim_tiempo, on='fecha_completa')

# Cálculos extendidos
fact_ventas['subtotal'] = fact_ventas['cantidad'] * fact_ventas['precio_unitario']
fact_ventas['impuesto_iva'] = fact_ventas['subtotal'] * 0.16
fact_ventas['monto_total'] = fact_ventas['subtotal'] + fact_ventas['impuesto_iva']

# Selección final de columnas para la Fact Table
fact_table = fact_ventas[[
    'id_transaccion', 
    'id_cliente_sk', 
    'id_producto_sk', 
    'id_tiempo_sk', 
    'cantidad', 
    'subtotal', 
    'impuesto_iva', 
    'monto_total'
]]

print("=== TABLA DE HECHOS (FACT_VENTAS - Muestra) ===")
print(fact_table.head())