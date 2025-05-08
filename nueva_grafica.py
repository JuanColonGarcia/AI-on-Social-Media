import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/predicciones.csv')

# Limpiamos columnas
df['titulo_procesado'] = df['titulo_procesado'].fillna('').astype(str)
df['texto'] = df['texto'].fillna('').astype(str)
df['prediccion_sentimiento'] = df['prediccion_sentimiento'].str.strip()
df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')

while True:
    # Input de búsqueda
    palabra_input = input("\n🔎 Escribe una palabra para buscar en los títulos: ").strip().lower()

    if not palabra_input:
        print("No escribiste ninguna palabra. Intenta de nuevo.")
        continue

    # Buscamos títulos que contengan esa palabra
    palabras_posibles = df['titulo_procesado'][df['titulo_procesado'].str.contains(palabra_input, case=False, regex=False)]

    # Extraemos palabras únicas que contengan el input
    palabras_unicas = set()
    for titulo in palabras_posibles:
        for palabra in titulo.split():
            if palabra_input in palabra.lower():
                palabras_unicas.add(palabra.lower())

    palabras_unicas = sorted(list(palabras_unicas))

    # Mostramos las sugerencias
    if not palabras_unicas:
        print(f"No se encontraron coincidencias para '{palabra_input}'.")
        continue
    else:
        print("\nPalabras encontradas:")
        for idx, palabra in enumerate(palabras_unicas):
            print(f"{idx + 1}. {palabra}")

    indices = input("\nEscribe los números de las palabras que quieres buscar (separados por comas): ")
    seleccion = []
    for i in indices.split(','):
        i = i.strip()
        if i.isdigit():
            idx = int(i) - 1
            if 0 <= idx < len(palabras_unicas):
                seleccion.append(palabras_unicas[idx])

    if not seleccion:
        print("No seleccionaste ninguna palabra válida. Intenta de nuevo.")
        continue

    print("\nPalabras seleccionadas:", seleccion)

    print("\n¿Dónde quieres buscar?")
    print("1. Solo en títulos")
    print("2. Solo en textos")
    print("3. En títulos y textos")
    opcion = input("👉 Escribe 1, 2 o 3: ")

    if opcion == '1':
        mask = df['titulo_procesado'].apply(lambda x: any(sel in x.lower() for sel in seleccion))
    elif opcion == '2':
        mask = df['texto'].apply(lambda x: any(sel in x.lower() for sel in seleccion))
    elif opcion == '3':
        mask = df.apply(lambda row: any(sel in row['titulo_procesado'].lower() or sel in row['texto'].lower() for sel in seleccion), axis=1)
    else:
        print("⚠️ Opción inválida. Se buscará solo en Títulos.")
        mask = df['titulo_procesado'].apply(lambda x: any(sel in x.lower() for sel in seleccion))

    df_filtrado = df[mask]

    # Mostramos los resultados
    if df_filtrado.empty:
        print("No se encontraron posts con esas palabras.")
    else:
        print(f"\nEncontrados {len(df_filtrado)} posts que contienen esas palabras.")

        sentimientos_count = df_filtrado['prediccion_sentimiento'].value_counts()

        plt.figure(figsize=(8, 6))
        sentimientos_count.plot(kind='bar', color=['blue', 'green', 'red'])

        plt.title(f"Distribución de sentimientos para {', '.join(seleccion)}", fontsize=14)
        plt.xlabel('Sentimiento', fontsize=12)
        plt.ylabel('Cantidad de Posts', fontsize=12)

        for i, v in enumerate(sentimientos_count):
            plt.text(i, v + 2, str(v), ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.show()

    otra = input("\n¿Quieres buscar otra palabra? (s/n): ").strip().lower()
    if otra != 's':
        print("\n ¡Hasta la próxima!")
        break
