import pandas as pd

# Cargamos el CSV corrigiendo líneas malas y las mostramos
df = pd.read_csv("data/nuevos_datos.csv", on_bad_lines='skip')
print("Nombres de columnas:", df.columns)

# Limpiamos posibles saltos raros en el 'titulo' y el 'texto'
df['titulo'] = df['titulo'].replace({'"': "'", '\n': ' ', '\r': ' '}, regex=True)
df['texto'] = df['texto'].replace({'"': "'", '\n': ' ', '\r': ' '}, regex=True)

# Quitamos filas que no tienen 'titulo' o 'texto' o que estén vacías
df = df.dropna(subset=['titulo', 'texto'])  
df = df[df['titulo'].str.strip() != '']  
df = df[df['texto'].str.strip() != ''] 

# Revisamos que las columnas necesarias existan 
df = df[['titulo', 'texto', 'score', 'fecha']]  

# Verificamos que las fechas estén en formato epoch timestamp y si están como números las convertimos datetime
if df['fecha'].dtype == 'float64':  
    df['fecha'] = pd.to_datetime(df['fecha'], unit='s', errors='coerce') 
else:
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')  

# Eliminamos filas donde la fecha no tiene sentido 
df = df.dropna(subset=['fecha'])

# Tomamos solo las primeras 1000 filas válidas
df = df.head(5000)

# CSV limpio
df.to_csv("data/nuevos_datos_limpios.csv", index=False)