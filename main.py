"""
Autor : Ing. Jesus Parra
Año 2024
"""
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

# URL de Datasets
ruta_developer = 'https://github.com/ing-jhparra/Sistema-de-Recomendacion-de-Videojuegos-para-Usuarios/blob/23fca09336a144d889c7895cc55777af12ea11fc/Datasets/developer.parquet'
ruta_user_items = 'https://github.com/ing-jhparra/Sistema-de-Recomendacion-de-Videojuegos-para-Usuarios/blob/4470d8eb352604a814323e260b97f66c71c8ba5b/Datasets/users_items_30mil.parquet'
ruta_user_reviews = 'https://github.com/ing-jhparra/Sistema-de-Recomendacion-de-Videojuegos-para-Usuarios/blob/63c02be8130aacc4fb995e5608f4c0b8febe3a7e/Datasets/user_review.parquet'

# Abrir y cargar Dataset a un dataframe
df_developer = pd.read_parquet(ruta_developer + '?raw=True', engine='auto')
df_user_items = pd.read_parquet(ruta_user_items + '?raw=True', engine='auto')
df_user_review = pd.read_parquet(ruta_user_reviews + '?raw=True', engine='auto')

@app.get('/', tags=['inicio'])
async def inicio():
    cuerpo = '<center><h1 style="background-color:#daecfe;">Proyecto Individual Numero 1:<br>Machine Learning Operations (MLOps)</h1></center>'
    return HTMLResponse(cuerpo)

@app.get("/developer/{desarrollador}",  tags=['developer'])
async def developer(desarrollador):
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

@app.get("/userdata/{user_id}",  tags=['userdata'])
async def userdata(user_id):

    '''
    Devuelve la cantidad de dinero gastado por el usuario, el porcentaje de recomendación y cantidad de items
             
    Parametro
    ---------
    str
        user_id : Identificador del usuario.
    
    Retorna
    -------
        dict: Diccionario 
              
              Cantidad Dinero          : Cantidad de dinero gastado.
              Porcentaje Recomendacion : Porcentaje de recomendaciones.
              Total de Items           : Cantidad de items.
    '''
    los_juegos = df_developer[['item_id','price']]
    el_usuario = df_user_review[df_user_review['user_id'] == user_id]
    recomendado = round(el_usuario[el_usuario["recommend"] == True].count() / (el_usuario[el_usuario["recommend"]==True].count() + 
                                                                            el_usuario[el_usuario["recommend"]==False].count()) * 100,2).iloc[0]
    
    los_items = df_user_items[df_user_items['user_id'] == user_id]
    los_items = los_items.merge(los_juegos, on = 'item_id',  how='left').fillna(0.0)
    los_items = los_items.groupby('user_id').agg({'item_id':'sum','price':'sum'}).reset_index()

    claves = ["Usuario", "Dinero gastado", "Porcentaje de recomendación", "Cantidad"]
    valor = [los_items["user_id"].iloc[0],los_items['price'].iloc[0], recomendado, int(los_items["item_id"].iloc[0])]

    diccionario = dict(zip(claves,valor))

    return "No existen registros" if len(los_juegos) == 0 else diccionario