"""
Estudiante : Jesus H. Parra B.
Academia : Henry
Carrera : Data Science 
Direccion Web : https://www.soyhenry.com/
email : parra.jesus@gmail.com
Año 2024
"""
# LIbrerias necesaria

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Rutas de Datasets que utilizara los endpoints

ruta_developer = r'Datasets/developer.parquet'
ruta_user_items = r'Datasets/users_items.parquet'
ruta_user_reviews =  r'Datasets/user_review.parquet'
ruta_sentiment_analysis =  r'Datasets/sentiment_analysis.parquet'
ruta_endpoint3 = r'Datasets/endpoint3.parquet'
ruta_endpoint4 = r'Datasets/los_mejores.parquet'

# Abrir y cargar Dataset para ser utilizados por los endpoints

df_developer = pd.read_parquet(ruta_developer, engine='auto')
df_user_items = pd.read_parquet(ruta_user_items, engine='auto')
df_user_review = pd.read_parquet(ruta_user_reviews, engine='auto')
df_sentiment_analysis = pd.read_parquet(ruta_sentiment_analysis, engine='auto')
horas_acumuladas = pd.read_parquet(ruta_endpoint3, engine='auto')
los_mejores = pd.read_parquet(ruta_endpoint4, engine='auto')

# Declaracion y definicion del modelo similitud del coseno para Machine Learning
# En este ejercicio me base en el siguiente video https://www.youtube.com/watch?v=7nago29IlxM&t=149s

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_sentiment_analysis["review"])
    
feactures =  np.column_stack([tfidf_matrix.toarray(), df_sentiment_analysis["recommend"], df_sentiment_analysis["sentiment_analysis"]])

similarity_matrix = cosine_similarity(feactures)

df_sentiment_analysis = df_sentiment_analysis.reset_index(drop=True)

# Declaracion y definicion de la clase FastAPI
app = FastAPI( 
    title = 'Machine Learning Operations (MLOps)',
    description='API para realizar consultas',
    version='1.0 / Jesus Parra (2024)'
)

# Se muestra en la siguiente ruta un sencillo titulo http://127.0.0.1:8000/
@app.get('/', tags=['inicio'])
async def inicio():
    '''
                    <strong>PROYECTO INDIVIDUAL Nº1</strong> <br>
                    <strong>Machine Learning Operations (MLOps)</strong>

    <strong>Alumno</strong>      : Jesus H. Parra B.<br>
    <strong>Carrera</strong>     : Ciencia de Datos<br>
    <strong>Cohorte</strong>     : 22<br>
    <strong>Año</strong>         : 2024  
    <strong>Correo</strong>                        : parra.jesus@gmail.com            
    '''
    cuerpo = '<center><h1 style="background-color:#daecfe;">Proyecto Individual Numero 1:<br>Machine Learning Operations (MLOps)</h1></center>'
    return HTMLResponse(cuerpo)

# Endpoint http://127.0.0.1:8000/developer/{desarrollador} 
@app.get("/developer/{desarrollador}",  tags=['developer'])
async def developer(desarrollador : str):
    '''
    <strong>Devuelve un diccionario año, cantidad de items y porcentaje de contenido libre por empresa desarrolladora</strong>
             
    Parametro
    ---------  
            desarrollador : Nombre de la empresa desarrolladora
    
    Retorna
    -------
            Anio                         : Año
            Cantidad Items               : Videos juegos desarrollados
            Porcentaje de contenido Free : Porcetnaje de contenidos gratuito
    '''
    
    lista_diccioanario = {"Anio" : list(),"Cantidad de items" : list(),"Porcentaje de contenido Free" : list()}

    # Filtramos por desarrollador
    el_desarrollador = df_developer[df_developer['developer'] == desarrollador]
    # Calcula el total de items por año
    cantidad_items = el_desarrollador.groupby('year')['item_id'].count()
    # Calcula el total de contenido gratis por año
    cantidad_gratis = el_desarrollador[el_desarrollador['price'] == 0.0].groupby('year')['item_id'].count()
    # Calcula el porcentaje de contenido gratis por año
    porcentaje_gratis = (cantidad_gratis / cantidad_items * 100).fillna(0).astype(int)
    # Damos formato para el retorno de la informacion
    for year, item_id_counts in cantidad_items.items():
        lista_diccioanario["Anio"].append(year)
        lista_diccioanario["Cantidad de items"].append(item_id_counts)
    for year, item_porc in porcentaje_gratis.items():
        lista_diccioanario["Porcentaje de contenido Free"].append(item_porc)
    
    diccionario = pd.DataFrame(lista_diccioanario).to_dict(orient='records')
    
    return  "No existen registros" if len(el_desarrollador) == 0 else diccionario

# Endpoint http://127.0.0.1:8000/userdata/{user_id} 
@app.get("/userdata/{user_id}",  tags=['userdata'])
async def userdata(user_id : str):

    '''
    <strong>Devuelve la cantidad de dinero gastado por el usuario, el porcentaje de recomendación y cantidad de items</strong>
             
    Parametro
    ---------
              user_id : Identificador unico del usuario.
    
    Retorna
    -------   
              Cantidad Dinero          : Cantidad de dinero gastado.
              Porcentaje Recomendacion : Porcentaje de recomendaciones.
              Total de Items           : Cantidad de items.
    '''
    try:
        los_juegos = df_developer[['item_id','price']]
        el_usuario = df_user_review[df_user_review['user_id'] == user_id]
    except :
        return 'Ocurrio un error para este usuario'
    recomendado = round(el_usuario[el_usuario["recommend"] == True].count() / (el_usuario[el_usuario["recommend"]==True].count() + 
                                                                            el_usuario[el_usuario["recommend"]==False].count()) * 100,2).iloc[0]
    
    los_items = df_user_items[df_user_items['user_id'] == user_id]
    los_items = los_items.merge(los_juegos, on = 'item_id',  how='left').fillna(0.0)
    los_items = los_items.groupby('user_id').agg({'item_id':'sum','price':'sum'}).reset_index()

    claves = ["Usuario", "Dinero gastado", "Porcentaje de recomendación", "Cantidad"]
    valor = [los_items["user_id"].iloc[0],los_items['price'].iloc[0], recomendado, int(los_items["item_id"].iloc[0])]

    diccionario = dict(zip(claves,valor))

    return "No existen registros" if len(los_juegos) == 0 else diccionario

# Endpoint http://127.0.0.1:8000/UserForGenre/{genero} 
@app.get("/UserForGenre/{genero}",  tags=['UserForGenre'])
def UserForGenre(genero : str) :
    '''
    <strong>Devuelve el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento</strong>
             
    Parametro
    ---------
              genero : genero del juego
    
    Retorna
    -------   
              Usuario          : Usuario con mas horas jugadas.
              Horas jugadas : Lista de diccionario Años vs Horas.
              
    '''
    horas_juagadas = horas_acumuladas[horas_acumuladas["genres"]==genero]
    usuarios = horas_juagadas.groupby(["user_id","year"]).agg({'horas_juego': 'sum'}).reset_index()
    jugador_mayor = horas_juagadas["user_id"].loc[0]
    x_anio = usuarios[usuarios['user_id'] == jugador_mayor][["year","horas_juego"]]
    x_anio.rename(columns={'year':'Año','horas_juego':'Tiempo'}, inplace=True)
    anio_juego= x_anio.to_dict('records')
    diccionario = {"Usuario con mas horas jugadas":jugador_mayor, "Horas jugadas" : anio_juego}

    return "No existen registros" if len(horas_juagadas) == 0 else diccionario

# Endpoint http://127.0.0.1:8000/UserForGenre/{genero} 
@app.get("/best_developer_year/{anio}",  tags=['best_developer_year'])
def best_developer_year(anio : int ) :
    '''
    <strong>Devuelve el top 3 de desarrolladores con juegos MÁS recomendados por usuarios para el año dado</strong>
             
    Parametro
    ---------
              año : Un año. Ejemplo 2024
    
    Retorna
    -------   
              Top 1 : Empresa Desarrolladora, Juegos mas recomendados
              Top 2 : Empresa Desarrolladora, Juegos mas recomendados
              Top 3 : Empresa Desarrolladora, Juegos mas recomendados
    '''
    juegos=los_mejores
    print(len(juegos))
    juegos=juegos[juegos['year'] == anio]
    reviews_filter=juegos.groupby(['developer']).agg({'sentiment_analysis':'sum'}).reset_index()
    reviews_filter= reviews_filter.sort_values(by= 'sentiment_analysis', ascending= False)
    desarrolladores =list(reviews_filter.iloc[:,0])
    positivos =list(reviews_filter.iloc[:,1])
    diccionario = {"Top 1":[{'Developer': desarrolladores[0]},{'Reviews positive':positivos[0]}], "Top 2":[{'Developer': desarrolladores[1]},{'Reviews positive':positivos[1]}], "Top 3":[{'Developer': desarrolladores[2]},{'Reviews positive':positivos[2]}]}

    return "No existen registros" if len(juegos) == 0 else diccionario

# Endpoint http://127.0.0.1:8000/recomendacion_juego/{item_id}
@app.get("/recomendacion_juego/{item_id}",  tags=['recomendacion'])
async def recomendacion_juego (item_id : int):
    '''
    <strong>Devuelve una cantidad de 5 juegos recomendado a partir del identifcador de un juego</strong>
             
    Parametro
    ---------
             item_id : Identificador unico del juego.
    
    Retorna
    -------   
             Diccionario con una lista de 5 juegos similiares recomendados a partir del ingresado
    '''

    producto = df_sentiment_analysis[df_sentiment_analysis['item_id'] == item_id]
    if not producto.empty:
        product_index = producto.index[0]
        product_similarities = similarity_matrix[product_index]
        most_similar_products_indices = np.argsort(-product_similarities)
        most_similar_products = df_sentiment_analysis.loc[most_similar_products_indices, 'item_name']
    else:
        return "Producto no encontrado"
    
    diccionario = {"Juegos recomendados" : list()}
    similares = most_similar_products[:5]
    diccionario["Juegos recomendados"] = [similar for similar in similares]
    diccionario

    return diccionario

# Endpoint http://127.0.0.1:8000/recomendacion_usuario/{user_id}
@app.get("/recomendacion_usuario/{user_id}",  tags=['recomendacion'])
async def recomendacion_usuario (user_id):
    '''
    <strong>Devuelve una cantidad de 5 juegos recomendado a partir del identifcador unico de un usuario</strong>
             
    Parametro
    ---------
             user_id : Identificador unico del juego.
    
    Retorna
    -------   
             Diccionario con una lista de 5 juegos similares recomendados por un usuario
    '''

    producto = df_sentiment_analysis[df_sentiment_analysis['user_id'] == user_id]
    if not producto.empty:
        product_index = producto.index[0]
        product_similarities = similarity_matrix[product_index]
        most_similar_products_indices = np.argsort(-product_similarities)
        most_similar_products = df_sentiment_analysis.loc[most_similar_products_indices, 'item_name']
    else:
        return "Producto no encontrado"
    
    diccionario = {"Juegos recomendados" : list()}
    similares = most_similar_products[:5]
    diccionario["Juegos recomendados"] = [similar for similar in similares]
    diccionario

    return diccionario