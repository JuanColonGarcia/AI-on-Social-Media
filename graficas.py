import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
df = pd.read_csv('data/predicciones.csv')

# Asegurar que no haya espacios raros
df['prediccion_sentimiento'] = df['prediccion_sentimiento'].str.strip()

# Convertir fecha
df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')

# Filtrar últimos 5 años
fecha_actual = pd.to_datetime('today')
fecha_inicio = fecha_actual - pd.DateOffset(years=5)
df_filtrado = df[df['fecha'] >= fecha_inicio]

# Crear columna 'mes'
df_filtrado['mes'] = df_filtrado['fecha'].dt.to_period('M').dt.to_timestamp()

# Agrupar por mes y sentimiento
sentimiento_por_mes = df_filtrado.groupby(['mes', 'prediccion_sentimiento']).size().reset_index(name='cantidad')

# Verificar si hay datos
if sentimiento_por_mes.empty:
    print("No hay datos agrupados por mes y sentimiento.")
else:
    # Definir colores personalizados
    palette = {
        'Neutral': 'blue',
        'Negativo': 'red',
        'Positivo': 'green'
    }

    # Plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        x='mes',
        y='cantidad',
        hue='prediccion_sentimiento',
        data=sentimiento_por_mes,
        marker='o',
        palette=palette
    )

    # Formato del gráfico
    plt.title('Evolución del Sentimiento hacia IA en Reddit (últimos 5 años) - Mensual')
    plt.xlabel('Mes')
    plt.ylabel('Cantidad de Posts')
    plt.xticks(
        ticks=sentimiento_por_mes['mes'].unique(),
        labels=[x.strftime('%b-%Y') for x in sentimiento_por_mes['mes'].unique()],
        rotation=45
    )
    
    plt.tight_layout()
    plt.show()

# Contar la cantidad de posts por sentimiento
sentimientos_count = df['prediccion_sentimiento'].value_counts()

# Crear un gráfico de barras con los resultados
plt.figure(figsize=(8, 6))
sentimientos_count.plot(kind='bar', color=['gray', 'green', 'red'])

# Agregar título y etiquetas
plt.title('Distribución de Sentimientos hacia IA en Reddit', fontsize=14)
plt.xlabel('Sentimiento', fontsize=12)
plt.ylabel('Cantidad de Posts', fontsize=12)

# Mostrar las cantidades en la parte superior de las barras
for i, v in enumerate(sentimientos_count):
    plt.text(i, v + 2, str(v), ha='center', va='bottom', fontsize=12)

# Mostrar el gráfico
plt.tight_layout()
plt.show()


# Verificar el rango de fechas
print("Rango de fechas en el DataFrame:", df['fecha'].min(), "a", df['fecha'].max())
