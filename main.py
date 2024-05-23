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

url_developer = 'https://github.com/ing-jhparra/Sistema-de-Recomendacion-de-Videojuegos-para-Usuarios/blob/23fca09336a144d889c7895cc55777af12ea11fc/Datasets/developer.parquet?raw=True'

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