"""
Autor : Ing. Jesus Parra
Año 2024
"""
import __main__
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# Instanciamos la clase FastAPI
app = FastAPI( 
    title = 'Machine Learning Operations (MLOps)',
    description='API para realizar consultas',
    version='1.0 / Jesus Parra (2024)'
)

#ruta = ".\\Datasets\\"
#datasets = ["steam_games.parquet", "users_items.parquet", "user_review.parquet", "developer.parquet" ]
url_developer = 'https://github.com/ing-jhparra/Sistema-de-Recomendacion-de-Videojuegos-para-Usuarios/blob/23fca09336a144d889c7895cc55777af12ea11fc/Datasets/developer.parquet?raw=True'

df_game = pd.read_parquet(ruta + datasets[0])
df_items = pd.read_parquet(ruta + datasets[1])
df_reviews = pd.read_parquet(ruta + datasets[2])
df_developer = pd.read_parquet(url_developer,engine='auto')


@app.get('/', tags=['inicio'])
async def inicio():
    cuerpo = '<center><h1 style="background-color:#daecfe;">Proyecto Individual Numero 1:<br>Machine Learning Operations (MLOps)</h1></center>'
    return HTMLResponse(cuerpo)

"""
Endpoint 1 : Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora.
"""

@app.get("/developer/{desarrollador}",  tags=['developer'])
def developer(desarrollador):
    '''
    Devuelve por año, cantidad de items y porcentaje de contenido libre por empresa desarrolladora
    
    Parametro
    ---------
    str
        desarrollador : Nombre de la empresa desarrolladora
    
    Retorna
    -------
    dict : Diccionario 
               Cantidad Items : Videos juegos desarrollados por año
               Contenido Free : Contenidos gratuito pro año
    '''
    # Filtramos por desarrollador
    filtrado_desarrollador = df_developer[df_developer['developer'] == desarrollador]
    # Calcula el total de items por año
    cantidad_items = filtrado_desarrollador.groupby('year')['item_id'].count()
    # Calcula el total de contenido gratis por año
    cantidad_gratis = filtrado_desarrollador[filtrado_desarrollador['price'] == 0.0].groupby('year')['item_id'].count()
    # Calcula el porcentaje de contenido gratis por año
    porcentaje_gratis = (cantidad_gratis / cantidad_items * 100).fillna(0).astype(int)

    diccionario = {
        'Cantidad de items': cantidad_items.to_dict(),
        'Porcentaje de contenido Free': porcentaje_gratis.to_dict()
    }
    
    return diccionario

def userdata(user_id):
    '''
    Devuelve la cantidad de dinero gastado por el usuario, el porcentaje de recomendación y cantidad de items
             
    Parametro
    ---------
    str
        user_id : Identificador del usuario.
    
    Retorna
    -------
        dict: Diccionario 
              Cantidad Dinero : Cantidad de dinero gastado por el usuario.
              Porcentaje Recomendacion : Porcentaje de recomendaciones realizadas por el usuario.
            - 'total_items' (int): Cantidad de items que tiene el usuario.
    '''
    ruta_archivo = "https://github.com/ing-jhparra/Sistema-de-Recomendacion-de-Videojuegos-para-Usuarios/blob/475a358fd05c67d0b65d5d9962e1eb6aa785ce57/Datasets/userdata.parquet?raw=True"
    df_userdata = pd.read_parquet(ruta_archivo, engine='auto')
    # Filtra por el usuario de interés
    usuario = df_userdata[df_userdata['user_id'] == user_id]
    # Calcula la cantidad de dinero gastado para el usuario de interés
    cantidad_dinero = df_gastos_items[df_gastos_items['user_id']== user_id]['price'].iloc[0]
    # Busca el count_item para el usuario de interés    
    count_items = df_gastos_items[df_gastos_items['user_id']== user_id]['items_count'].iloc[0]
    
    # Calcula el total de recomendaciones realizadas por el usuario de interés
    total_recomendaciones = usuario['reviews_recommend'].sum()
    # Calcula el total de reviews realizada por todos los usuarios
    total_reviews = len(df_reviews['user_id'].unique())
    # Calcula el porcentaje de recomendaciones realizadas por el usuario de interés
    porcentaje_recomendaciones = (total_recomendaciones / total_reviews) * 100
    
    return {
        'cantidad_dinero': int(cantidad_dinero),
        'porcentaje_recomendacion': round(float(porcentaje_recomendaciones), 2),
        'total_items': int(count_items)
    }