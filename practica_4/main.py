
from faker import Faker
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuracion para obtener resultados reproducibles
fake = Faker("es_MX")
Faker.seed(42)
np.random.seed(42)

# 1. Generar 1000 registros sinteticos
registros = []

for i in range(1000):
    registros.append({
        "id_cliente": i + 1,
        "nombre": fake.name(),
        "email": fake.email(),
        "direccion": fake.address().replace("\n", ", "),
        "telefono": fake.phone_number(),
        "estado_civil": np.random.choice(
            ["Soltero", "Casado", "Divorciado"],
            p=[0.5, 0.4, 0.1]
        ),
        "ingreso_anual": round(
            np.random.uniform(100000, 1000000), 2
        ),
        "churn": np.random.choice([0, 1], p=[0.8, 0.2])
    })

# Crear el DataFrame
df_sintetico = pd.DataFrame(registros)

# 2. Separar caracteristicas (X) y variable objetivo (y)
X = df_sintetico.drop(
    columns=["id_cliente", "nombre", "email", "direccion",
             "telefono", "churn"]
)

y = df_sintetico["churn"]

# 3. Convertir estado civil a columnas numericas
X = pd.get_dummies(
    X, columns=["estado_civil"], dtype=int
)

# 4. Dividir los datos: 80% entrenamiento y 20% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Mostrar resultados
print("\n--- MUESTRA DE DATOS SINTETICOS ---")
print(df_sintetico.head())

print("\n--- DIMENSIONES ---")
print("Dataset completo:", df_sintetico.shape)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\n--- PROPORCION DE CHURN ---")
resultados = pd.DataFrame({
    "Conjunto": ["Completo", "Train", "Test"],
    "Registros": [len(y), len(y_train), len(y_test)],
    "Proporcion churn": [
        y.mean(), y_train.mean(), y_test.mean()
    ]
})
print(resultados.to_string(index=False))

# 6. Guardar los archivos generados
df_sintetico.to_csv("clientes_sinteticos.csv", index=False)
resultados.to_csv("resultados_churn.csv", index=False)

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\nArchivos CSV guardados correctamente.")
