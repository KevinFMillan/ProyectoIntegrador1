from fastapi import FastAPI
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_parquet("../data/dataset_movies.parquet")
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

@app.get("/recomendacion")
def recomendacion(titulo):
    vectorizador = TfidfVectorizer()
    dataVectorizada = vectorizador.fit_transform(data['overview'])
    
    carac = np.column_stack([dataVectorizada.toarray()])
    return f"en base a la pelicula {titulo} te recomendamos que veas estas otras 5 peliculas:"