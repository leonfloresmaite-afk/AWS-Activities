import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Generacion del conjunto de datos con ruido
data = {
    'edad': [25, 30, np.nan, 45, 22, 120, 35, 28, np.nan, 40],
    'ingreso_mensual': [15000, 22000, 18000, np.nan, 12000, 950000, 25000, 21000, 19000, np.nan],
    'scoring_credito': [650, 700, 680, 710, 590, 800, 670, 640, 620, 730]
}

df_dirty = pd.DataFrame(data)
# Guardar una copia de los datos originales
df_original = df_dirty.copy()
print("=== DATASET ORIGINAL ===")
print(df_dirty)

print("\n=== DATOS FALTANTES ===")
print(df_dirty.isnull().sum())

# 1. Detección de valores atípicos mediante IQR

Q1 = df_dirty['ingreso_mensual'].quantile(0.25)
Q3 = df_dirty['ingreso_mensual'].quantile(0.75)

IQR = Q3 - Q1

limite_superior = Q3 + 1.5 * IQR

print("\n=== DETECCION DE VALORES ATIPICOS ===")
print(f"Primer cuartil (Q1): ${Q1:,.2f}")
print(f"Tercer cuartil (Q3): ${Q3:,.2f}")
print(f"Rango intercuartil (IQR): ${IQR:,.2f}")
print(f"Limite superior: ${limite_superior:,.2f}")

print("\nIngresos que superan el limite:")
print(df_dirty[df_dirty['ingreso_mensual'] > limite_superior])


# 2. Reemplazo de valores atipicos por NaN

df_dirty.loc[
    df_dirty['ingreso_mensual'] > limite_superior,
    'ingreso_mensual'
] = np.nan

df_dirty.loc[df_dirty['edad'] > 100, 'edad'] = np.nan

print("\n=== DATASET DESPUES DE TRATAR LOS VALORES ATIPICOS ===")
print(df_dirty)

print("\n=== TOTAL DE VALORES FALTANTES DESPUES DEL TRATAMIENTO ===")
print(df_dirty.isnull().sum())


# 3. Imputacion de valores faltantes con KNNImputer

imputer = KNNImputer(n_neighbors=2)

df_clean_array = imputer.fit_transform(df_dirty)

df_clean = pd.DataFrame(
    df_clean_array,
    columns=df_dirty.columns
)

print("\n=== DATASET LIMPIO E IMPUTADO ===")
print(df_clean)

print("\n=== VERIFICACION DE VALORES FALTANTES ===")
print(df_clean.isnull().sum())


# 4. Graficas antes y despues del tratamiento de outliers

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Grafica de los datos originales
axes[0].boxplot(df_original['ingreso_mensual'].dropna())
axes[0].set_title('Antes de la limpieza')
axes[0].set_ylabel('Ingreso mensual')

# Grafica despues del tratamiento de outliers
axes[1].boxplot(df_dirty['ingreso_mensual'].dropna())
axes[1].set_title('Despues del tratamiento IQR')
axes[1].set_ylabel('Ingreso mensual')

plt.tight_layout()
plt.savefig('comparacion_outliers.png')
plt.show()


# 5. Comparacion entre KNN, media y mediana

from sklearn.impute import SimpleImputer

# Comparar diferentes cantidades de vecinos
knn_2 = KNNImputer(n_neighbors=2)
resultado_knn_2 = pd.DataFrame(
    knn_2.fit_transform(df_dirty),
    columns=df_dirty.columns
)

knn_3 = KNNImputer(n_neighbors=3)
resultado_knn_3 = pd.DataFrame(
    knn_3.fit_transform(df_dirty),
    columns=df_dirty.columns
)

# Imputacion usando la media
imputer_media = SimpleImputer(strategy='mean')
resultado_media = pd.DataFrame(
    imputer_media.fit_transform(df_dirty),
    columns=df_dirty.columns
)

# Imputacion usando la mediana
imputer_mediana = SimpleImputer(strategy='median')
resultado_mediana = pd.DataFrame(
    imputer_mediana.fit_transform(df_dirty),
    columns=df_dirty.columns
)

# Mostrar comparacion de los registros que tenian datos faltantes
print("\n=== COMPARACION DE METODOS ===")

filas_faltantes = df_dirty[
    df_dirty.isnull().any(axis=1)
].index

print("\n--- KNN (k=2) ---")
print(resultado_knn_2.loc[filas_faltantes])

print("\n--- KNN (k=3) ---")
print(resultado_knn_3.loc[filas_faltantes])

print("\n--- MEDIA ---")
print(resultado_media.loc[filas_faltantes])

print("\n--- MEDIANA ---")
print(resultado_mediana.loc[filas_faltantes])

