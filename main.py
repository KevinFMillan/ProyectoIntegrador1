from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

peliculas = pd.read_parquet("./data/dataset_movies.parquet")
app = FastAPI()

@app.get("/mes")
def cantidad_filmaciones_mes(mes: str):
    return f"x cantidad de peliculas fueron estrenadas en el mes de {mes}"

@app.get("/dia")
def cantidad_filmaciones_dia(dia: str):
    return f"x cantidad de peliculas fueron estrenadas en los dias {dia}"

@app.get("/score")
def score_titulo(titulo_de_la_filmacion: str):
    return f"la pelicula {titulo_de_la_filmacion} fue estrenada en el anio xxxx con un score/popularidad de xxxx"

@app.get("/votos")
def votos_titulo(titulo_de_la_filmacion: str):
    return f"la pelicula {titulo_de_la_filmacion} fue estrenada en el anio xxxx. La misma cuenta con un total de xxxx valoraciones, con un promedio de xxxx"

@app.get("/actor")
def get_actor(nombreActor: str):
    return f"El actor {nombreActor} ha participado de xxxx cantidad de filmaciones, el mismo ha conseguido un retorno de xxxx con un promedio de xxxx por filmacion"

@app.get("/director")
def get_director(nombeDirector: str):
    return f"{nombeDirector}"

vectorizer = TfidfVectorizer(stop_words='english')
overviews_tokenizados = vectorizer.fit_transform(peliculas['overview'])

@app.get("/recomendacion")
def recomendacion(titulo: str):
    pelicula = peliculas[peliculas['title'] == titulo]
    if pelicula.empty:
        return {"error": "Película no encontrada"}
    
    # Calcular la similitud del coseno con todas las demás peliculas
    similitudes = cosine_similarity(overviews_tokenizados[peliculas.index == pelicula.index[0]], overviews_tokenizados)
    
    # Obtener las 5 peliculas más similares
    índices_similares = similitudes[0].argsort()[-6:-1][::-1]  # Ignorar la misma película y ordenar
    
    # Retornar los titles de las peliculas más similares
    return peliculas.iloc[índices_similares]['title'].tolist()
titulo = 'Toy Story'
peliculas_similares = recomendacion(titulo)
print(peliculas_similares)