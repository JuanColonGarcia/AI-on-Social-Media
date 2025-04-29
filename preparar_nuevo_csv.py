import pandas as pd

# Cargar el CSV corrigiendo líneas malas
df = pd.read_csv("data/nuevos_datos.csv", on_bad_lines='skip')

# Inspeccionar las columnas
print("Nombres de columnas:", df.columns)

# Limpiar posibles saltos raros en 'titulo' y 'texto'
df['titulo'] = df['titulo'].replace({'"': "'", '\n': ' ', '\r': ' '}, regex=True)
df['texto'] = df['texto'].replace({'"': "'", '\n': ' ', '\r': ' '}, regex=True)

# Quitar filas que no tienen 'titulo' o 'texto' o que estén vacías
df = df.dropna(subset=['titulo', 'texto'])  # Asegurarse de que 'titulo' y 'texto' no sean NaN
df = df[df['titulo'].str.strip() != '']  # Eliminar filas donde 'titulo' esté vacío
df = df[df['texto'].str.strip() != '']  # Eliminar filas donde 'texto' esté vacío

# Revisar que las columnas necesarias existan y corregir si es necesario
df = df[['titulo', 'texto', 'score', 'fecha']]  # Solo mantener las columnas necesarias

# Inspeccionar las primeras filas de la columna 'fecha' antes de la conversión
print("\nPrimeras fechas antes de la conversión:\n", df['fecha'].head())

# Verificar si las fechas están en formato epoch timestamp
if df['fecha'].dtype == 'float64':  # Si las fechas están como números (epoch timestamp)
    df['fecha'] = pd.to_datetime(df['fecha'], unit='s', errors='coerce')  # Convertir a datetime
else:
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')  # Si ya está en formato fecha, convertirlo

# Eliminar filas donde la fecha no tiene sentido (NaT)
df = df.dropna(subset=['fecha'])

# Tomar solo las primeras 1000 filas válidas
df = df.head(1000)

# Guardar el CSV limpio
df.to_csv("data/nuevos_datos_limpios.csv", index=False)

print("✅ CSV limpio y funcional guardado como 'data/nuevos_datos_limpios.csv'")
