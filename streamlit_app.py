"""
    File: streamlit_app.py
    Author: Alejandro Bustamante
    Email: alejandro.bustamante@un.org
        
    Date: 10/10/2024
    Description: Script para visualización de producciones de publicacones CEPAL.
"""

import streamlit as st
import pandas as pd
import design
import plotly.express as px
import ast
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyvis.network import Network
import tempfile
from plotly.graph_objs import *
import os
import base64

# Librerias para chatbot
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
import langchain
import matplotlib.pyplot as plt
langchain.debug = True



####################################

st.set_page_config(
    layout= "wide",
    page_icon= "Image.png",
    page_title= "Agenda Investigación - CEPAL"
)


# Llamar los estilos personalizados
design.local_css("styles.css")

# Titulo
# =================================================================================
# Imagen
# =================================================================================

def st_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

st.markdown("""
    <style>
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header img {
        float: right;
        width: 300px; /* Adjust size */
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="header"><h1>Dashboard interactivo de agenda de investigación CEPAL</h1><img src="data:image/png;base64,' + 
            st_image_to_base64("Image.png") + '" alt="Header Image"></div>', 
            unsafe_allow_html=True)

st.write("Una iniciativa del Laboratorio de Prospectiva, Innovación e Inteligencia Artificial.")
path = os.path.dirname(__file__)

# =======================================================================================================
# BASE DE DATOS & TRANSFORMACIONES
# =======================================================================================================
excel_file = 'datos_full.xlsx' #Nombre archivo a importar  'xlsx' hace referencia a excel
sheet_name = 'Sheet1' #la hoja de excel que voy a importar

df_tot = pd.read_excel(excel_file, #importo el archivo excel
                   sheet_name = sheet_name) #le digo cual hoja necesito

df = df_tot#[df_tot['Sustantivo']=='SI']

division = df['Entidad'].sort_values().unique().tolist() # se crea una lista unica de la columna Division
#tipo = df['tipo_gr'].unique().tolist() # se crea una lista unica de la columna tipo de documento
#year = df['dc.year'].unique().tolist() # se crea una lista unica de la columna año

# Paso 1: Dividir los valores por comas
df['assigned_topics'] = df['assigned_topics'].str.split(',')

# Paso 2: Usar explode para que cada valor sea una fila diferente
df_temas = df.explode('assigned_topics')

# Eliminar espacios adicionales si es necesario
df_temas['assigned_topics'] = df_temas['assigned_topics'].str.strip()

df_temas['assigned_topics'] = df_temas['assigned_topics'].replace("INNOVACIÓN CIENCIA Y TECNOLOGÍA", "INNOVACIÓN, CIENCIA Y TECNOLOGÍA")



# BRECHAS

brechas = df_tot[df_tot['Brechas Atendidas'].notnull()]

# Paso 1: Dividir los valores por comas
brechas['brechas'] = brechas['Brechas Atendidas'].astype(str).str.split(',')

# Paso 2: Usar explode para que cada valor sea una fila diferente
brechas_exp = brechas.explode('brechas')

# Si es necesario, puedes convertir los valores de string a integer
brechas_exp['brechas'] = brechas_exp['brechas'].astype(int)


# ------------------------
# SECTORES 

sectores = df_tot[df_tot['Sectores Atendidos'].notnull()]

# Paso 1: Dividir los valores por comas
sectores['sectores'] = sectores['Sectores Atendidos'].astype(str).str.split(',')

# Paso 2: Usar explode para que cada valor sea una fila diferente
sectores_exp = sectores.explode('sectores')

# Si es necesario, puedes convertir los valores de string a integer
sectores_exp['sectores'] = sectores_exp['sectores'].astype(int)




# Para grupos temáticos realizamos la siguiente trasformación
#df = df.assign(temas=df['cepal.topicSpa'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x))
#df_temas = df.explode('temas')

# Lectura de la otra base de datos de clusters
clusters = pd.read_excel('Clusters.xlsx', sheet_name = 'g3')

# =======================================================================================================
# FILTRO
st.sidebar.title('Filtrar datos')
#Crear un slider de edad
#year_selector = st.sidebar.slider('Seleccione años a visualizar:',
#                        min_value = min(year), #el valor minimo va a ser el valor mas pequeño que encuentre dentro de la columna EDAD PERSONA ENCUESTADA
#                        max_value = max(year),#el valor maximo va a ser el valor mas grande que encuentre dentro de la columna EDAD PERSONA ENCUESTADA
#                        value = (min(year),max(year)),
#                        help = "Seleccione el rango de años para filtrar las publicaciones.") #que tome desde el minimo, hasta el maximo

#crear multiselectores
#Agregar opción Todas
division_all = ['Todas'] + division
#tipo_all = ['Todos'] + tipo

division_selector = st.sidebar.multiselect('Divisiones:',
                                division_all,
                                default = 'Todas')

#tipo_selector = st.sidebar.multiselect('Tipo de documentos:',
#                                    tipo_all,
#                                    default = 'Todos')

# Si "Todos" está seleccionada, seleccionar todas las demás divisiones
#if 'Todos' in tipo_selector:
#    tipo_selector = tipo

# Si "todas" está seleccionada, seleccionar todas las demás divisiones
if 'Todas' in division_selector:
    division_selector = division
# =======================================================================================================
# FILTROS DEL SELECT BOX
mask = (
#    df['dc.year'].between(*year_selector) & 
    df['Entidad'].isin(division_selector) #& 
#    df['tipo_gr'].isin(tipo_selector)
    )

# Debemos filtrar para que funcione
df_filtrado = df.loc[mask]

if df_filtrado.empty:
    st.warning("Por favor seleccione al menos un filtro para mostrar la información.")
    st.stop() 
# Agrupación por tipo
df_agrupado = df_filtrado.groupby(['Per']).count()[['ids']]
df_agrupado.reset_index(inplace= True)
df_agrupado.rename({'ids': 'Cantidad',
                            "Per": "Periodo"},axis= 1, inplace= True)
# Hacemos una agrupación ahora por división
div_agrupado = df_filtrado.groupby(['Entidad']).count()[['ids']] #que me agrupe por CALIFICACION y me cuente por los datos de  EDAD PERSONA ENCUESTADA
div_agrupado.reset_index(inplace = True)
div_agrupado.rename(columns= {"Entidad": "Tipo",
                              "ids": "Cantidad"}, inplace= True)
# Hacemos una agrupación ahora por temas
tem_agrupado = df_temas.groupby(['assigned_topics']).count()[['ids']] #que me agrupe por CALIFICACION y me cuente por los datos de  EDAD PERSONA ENCUESTADA
tem_agrupado.reset_index(inplace= True)
tem_agrupado.rename(columns= {"assigned_topics": "Tipo",
                              "ids": "Cantidad"}, inplace= True)

# Base de datos con los datos filtrados por división y por años
df_divisiones = df_filtrado.groupby(['Entidad'], as_index = False)['ids'].count() #hago un tipo de TABLA DINAMICA para agrupar los datos de una mejor manera, lo que hago aqui es que por cada EPS, me cuente la cantidad de personas encuestadas***
#df_years = df_filtrado.groupby(['dc.year'], as_index = False)['dc.identifier.uri'].count().rename(columns={'dc.identifier.uri': 'Cantidad'})

brechas_filt = brechas.loc[mask]

sectores_filt = sectores.loc[mask]

# =======================================================================================================
# Vamos a crear tabs para visualizar los distintos contenidos
publicaciones, g_tematicos, redes, brechas, sectores, chat = st.tabs(["Inicio",
                                                              "Temas",
                                                              "Redes",
                                                              "Brechas",
                                                              "Sectores",
                                                              "Chatbot"])

with publicaciones:
    st.title("Agenda futura de investigación CEPAL")
    st.write("""
             Esta herramienta recopila y sistematiza la agenda de investigación planteada por las divisiones, subsedes y oficinas
             en el marco del grupo de fuerza de tareas de invesitgaciones. A través de esta se puede explorar las investigaciones
             propuestas, la división autora, relación con visión institucional de CEPAL, tópicos clave, entre otros.
             La herramienta fue construida en base a la información recopilada y constituye un ejercicio piloto del laboratorio de
             prospectiva, innovación e inteligencia artificial de la CEPAL 
             """)                                      
    #Ahora necesito que esos selectores y slider me filtren la informacion
    numero_resultados = df[mask].shape[0] # number of availables rows

    st.markdown(f'*Resultados Disponibles: {numero_resultados}*') ## sale como un titulo que dice cuantos resultados tiene para ese filtro
    
    #pivot para reporte
    # Crear la pivot table
 #   pivot_table = pd.pivot_table(df_filtrado, 
 #                               index='tipo_gr',  # Columna para las filas
 #                               columns='dc.year',  # Columna para las columnas
 #                               values='dc.identifier.uri',  # Columna a contar
 #                               aggfunc='count',  # Función de agregación (en este caso, contar)
 #                               fill_value=0)  # Valor para los campos vacíos

    #st.dataframe(pivot_table)
    # Gráfico de líneas para la serie de tiempo
 #   line_chart = px.line(df_years,
 #                   title = 'Evolución de producción de documentos', 
 #                   x='dc.year',
 #                   y='Cantidad',
 #                   text ='Cantidad',
 #                   labels={'dc.year':'Año'}
 #                   )

    # Ajustar la posición del texto para que esté un poco por encima de cada punto
 #   line_chart.update_traces(textposition='top center')

 #   st.plotly_chart(line_chart, use_container_width= True) #mostrar el grafico de barras en streamlit

    # ordenamos los valores de mayor a menor
    df_divisiones.sort_values(by = "ids", inplace= True)
    # Calculamos el total de publicaciones por división
    total = df_divisiones['ids'].sum()
    df_divisiones['dc.identifier.uri.per']  = (df_divisiones['ids'] / total)

    plot = px.bar(df_divisiones,
                  title = "Líneas de investigación propuestas por división",
                  y = "Entidad",
                  x = "ids",
                  orientation= "h",
                  text_auto= '0',
                  height= 600, 
                  labels= {"Entidad":"División",
                            "ids":"Total de investigaciones"}
                            )
    #plot.update_layout(xaxis_tickformat = ".0")
    plot.update_traces(textfont_size = 12)
    st.plotly_chart(plot, use_container_width= True)
    
    #Crear un gráfico de barras
    bar_chart = px.bar(df_agrupado,
                       title = 'Publicaciones por tipo de documento',
                       x = "Periodo",
                       y = "Cantidad",
                       color = "Periodo",
                       barmode = "group",
                       text_auto= True,
                       labels = {"Tipo": "Tipo de documento",
                                 "dc.year": "Año"}
                        
    )




#    st.plotly_chart(bar_chart, use_container_width= True)

#    sel_tiempo = st.selectbox(label = "Seleccione el año:", options = tem_agrupado["Per"].unique())
#    top = tem_agrupado[tem_agrupado['Per'] == sel_tiempo].sort_values(by='Cantidad', ascending=False).head(15)

#    st.subheader(f'Evolución 15 temas más frecuentes en el año {sel_tiempo}')
#    bar_plot = px.bar(top, x = "Tipo",
#                       y = "Cantidad", height = 550, width= 800,
#                       text_auto= True,
#                         labels = {'dc.year':'Año'})
#    st.plotly_chart(bar_plot, use_container_width= True)

    st.write("<h2 style='font-weight: bold; font-size: 18px;'>Detalle de líneas de investigación</h2>", unsafe_allow_html=True)
    st.write("Para filtrar por división, subsede u oficina, seleccionar en el panel de la izquierda ")

    st.dataframe(
        df_filtrado[['Titulo','Detalle','Entidad']],
        column_config={
#            "dc.title": "Título",
            "Detalle": 'Preguntas de investigación',
            "Entidad": "División"        
        },
        hide_index=True,
        use_container_width= True
    )

with g_tematicos:
    st.title("Temas de investigación")
    st.write(
        """En esta sección se agrupan las líneas de investigación por temas a partir de los tópicos de CEPAL con los que se
        categorizan las publicaciones. En el primer gráfico cada punto representa un tópico y su posición vertical es la frecuencia
        del tópico en la agenda de investigación. El color se asocia a una dimensión del desarrollo y en el eje horizontal su afinidad
        a un subgrupo temático.

El segundo gráfico representa la misma frecuencia de los tópicos ordenados de mayor a menor. En la barra izquierda se puede
filtrar por una o más divisiones, subsedes y u oficinas para realizar análisis específico. En la parte inferior se puede 
obtener el listado de las investigaciones a partir de un tópico en particular.
        """
    )

    df_fil_tema = df_temas.loc[mask]
    tem_div = df_fil_tema.groupby(['assigned_topics']).size().reset_index(name = "frecuencia")

    tem_div = pd.merge(tem_div, clusters, left_on='assigned_topics', right_on='temas', how='left')

    

    numero_resultados = tem_div.shape[0] # number of availables rows
    st.markdown(f'*Total temas:{numero_resultados}*') # sale como un titulo que dice cuantos resultados tiene para ese filtro

    # Obtén la lista ordenada de clusters
    clusters_ordenados = sorted(tem_div['Cluster'].unique()) 

    # Guarda los colores originales  
    original_colors = {
        '1. Desarrollo económico': '#E41B1C',
        '2. Desarrollo social': '#377EB8',
        '3. Sustentabilidad ambiental y gestión de recursos naturales': '#4DAF4A',
        '4. Desarrollo productivo, innovación y aprovechamiento tecnológico': '#984EA3',
        '5. Institucionalidad, gobernanza y temas transversales': '#FF7F00'
    }
                    
    # Crear un DataFrame para cada categoría
    data_frames = {cluster: tem_div[tem_div['Cluster'] == cluster] for cluster in tem_div['Cluster'].unique()}

    # Crear una figura de dispersión para cada categoría
    graf = make_subplots(rows=1, cols=1, shared_xaxes=True, shared_yaxes=True)

    # Lista para mantener el orden de las trazas
    ordered_traces = []

    for cluster in clusters_ordenados:
        data_frame = data_frames[cluster]
        text_info = [
            f'Subject: {valor}<br>Publicaciones: {frec}' for valor, frec in zip(data_frame['temas'], 
                                                                                                data_frame['frecuencia']                                                                                             
                                                                            )
        ]
        trace = go.Scatter(
            x=data_frame['Item'], 
            y=data_frame['frecuencia'],
            mode='markers',
            name=cluster,
            text=text_info,
            hoverinfo='text',
            visible=True,  # Hacer visible por defecto
            marker=dict(
                color=original_colors[cluster],
                size=12,
                line=dict(
                    width=2,
                    color='white'
                ),
                opacity=0.6
            )
        )
        graf.add_trace(trace)
        ordered_traces.append(trace)

    # Configura la figura   
    graf.update_layout(
        title="Distribución Temática de la Investigación",
        height=800,
        width=800, 
        showlegend=True,
        legend=dict(
            x=0.5,
            y=-0.7,
            traceorder='normal',  # Ordenar por el orden de clusters_ordenados
            orientation='h',
            bgcolor='rgba(255, 255, 255, 0.6)',
            xanchor='center',  # Anclaje horizontal al centro
            yanchor='bottom',  # Anclaje vertical en la parte inferior
        )
    )

    # Configura la interacción con la leyenda
    graf.update_layout(legend=dict(itemclick='toggleothers', traceorder='normal'))


    st.plotly_chart(graf, use_container_width= True)


    # Gráfico de barras
    # Crear el gráfico interactivo
    tem_div = tem_div.sort_values(by = ["frecuencia","temas"], ascending = [False, True])
    sorted_temas =tem_div["temas"].tolist()
    bar = px.bar(tem_div, x='temas', y='frecuencia',
                  color = 'Cluster', color_discrete_map = original_colors,
                  opacity=0.6)

    # Ordenar los temas de mayor a menor frecuencia
    bar.update_xaxes(categoryorder='array', categoryarray = sorted_temas)

    # Personalizar el diseño del gráfico
    bar.update_layout(title='Frecuencia de temas en agenda de investigación',
                        yaxis_title='Frecuencia de tópicos en investigaciones propuestas',
                        xaxis_tickangle=-45,
                        width=800,
                        height=850,
                        showlegend=True,
                        legend=dict(
                            title='Grupo temático',
                            x=0.5,
                            y=-0.7,
                            orientation='h',
                            bgcolor='rgba(255, 255, 255, 0.6)',
                            xanchor='center',  # Anclaje horizontal al centro
                            yanchor='bottom'  # Anclaje vertical en la parte inferior
                            )
        )

    st.plotly_chart(bar, use_container_width= True)

    tema_df = tem_div.loc[tem_div['frecuencia'].idxmax(), 'temas']


    st.subheader('Detalles de agenda de investigación a partir de un tema seleccionado')


    tema = tem_div['temas'].unique().tolist() # se crea una lista unica de la columna Division

    tema_selector = st.selectbox('Seleccionar tema:', tema, index=tema.index(tema_df))


    # Crea un contenedor vacío para el dataframe
    df_container = st.empty()

    

    # Dentro del contenedor, crea el dataframe
    with df_container.container():
        # Código para crear el dataframe filtrado por tema_selector
        df_filtrado_tema = df_fil_tema[df_fil_tema['assigned_topics']==tema_selector][['Titulo','Detalle','assigned_topics','Periodo','Entidad']]
        st.write("Total: ", len(df_filtrado_tema))
        st.dataframe(df_filtrado_tema)

    # Actualiza el dataframe dentro del contenedor cuando el tema_selector cambia

    with df_container.container():
        df_filtrado_tema = df_fil_tema[df_fil_tema['assigned_topics']==tema_selector][['Titulo','Detalle','assigned_topics','Periodo','Entidad']]
        st.write("Total: ", len(df_filtrado_tema))  
        st.dataframe(df_filtrado_tema,
                    column_config={
      #      "dc.title": "Título",
      #      "dc.identifier.uri": st.column_config.LinkColumn("Enlace"),
      #      "dc.year": "Año",
      #      "division": "División"    
        },
        hide_index=True,
        use_container_width= True)

#with temas:
#    st.title("Temas")

#    st.write(
#    """Temas de publicaciones y divisiones autoras""")

#    def grafico_temas_division_interactivo(df, rango_temas=(0, 25), colores_division=None):

        # Calcular la frecuencia de cada tema y división
#        frecuencia = df_fil_tema.groupby(['assigned_topics', 'Entidad']).size().reset_index(name='frecuencia')

        # Obtener los temas ordenados por frecuencia
#        temas_ordenados = frecuencia.groupby('assigned_topics')['frecuencia'].sum().sort_values(ascending=False)

        # Obtener los temas dentro del rango especificado
#        temas_rango = temas_ordenados.reset_index()['assigned_topics'][rango_temas[0]:rango_temas[1]]

        # Filtrar los datos para los temas en el rango
#        frecuencia = frecuencia[frecuencia['assigned_topics'].isin(temas_rango)]

        # Crear un diccionario que mapee cada división a un color específico
        colores_division_dict = {'ASUNTOS DE GÉNERO': '#E999FF','Brasilia': '#24A215', 'Bogotá': '#968783',
                                'Buenos Aires': '#A0D1F8','CEPAL':'#1698FE','COMERCIO INTERNACIONAL E INTEGRACIÓN':'#104E7F',
                                'DESARROLLO ECONÓMICO':'#F50D2D','DESARROLLO PRODUCTIVO Y EMPRESARIAL':'#8D29E1',
                                'DESARROLLO SOCIAL':'#060DF7','DESARROLLO SOSTENIBLE Y ASENTAMIENTOS HUMANOS':'#88FF01',
                                'ESTADÍSTICAS':'#FF7012','Interdivisional':'#514F4B','México': '#8CFFAC', 'Montevideo': '#F7E8AD',
                                'PLANIFICACIÓN PARA EL DESARROLLO': '#F3FA0B', 'POBLACIÓN Y DESARROLLO': '#FFC300',
                                'RECURSOS NATURALES':'#8A4B2E','Revista':'#FF4BC9','Puerto España':'#EC936F',
                                'Washington': '#DFDDDC',
                                }

        # Crear el gráfico interactivo
       # fig = px.bar(frecuencia, x='assigned_topics', y='frecuencia', color='Entidad', barmode='stack',
       #             color_discrete_map=colores_division_dict)

        # Ordenar los temas de mayor a menor frecuencia
       # fig.update_xaxes(categoryorder='total descending')

        # Personalizar el diseño del gráfico
       # fig.update_layout(title='Frecuencia de temas por unidad organizacional',
       #                 xaxis_title='Temas',
       #                 yaxis_title='Cantidad de publicaciones',
       #                 legend_title='Unidad organizacional',
       #                 xaxis_tickangle=-45,
       #                 width=1100,
       #                 height=800)

        # Mostrar el gráfico y conteo
        #numero_resultados = temas_ordenados.shape[0] ##number of availables rows
        #st.markdown(f'*Total temas:{numero_resultados}*') ## sale como un titulo que dice cuantos resultados tiene para ese filtro

       # st.plotly_chart(fig, use_container_width= True)

    
    # Ejemplo de uso
   # grafico_temas_division_interactivo(df, rango_temas=(0, 151))


with redes:
    st.title("Potenciales redes de colaboración")
    st.write("""
             Esta visualización permite observar las potenciales redes de colaboración entre divisiones a partir de los temas abordados
             en sus investigaciones.
             Los triángulos representan a las unidades organizacionales (divisiones, subsedes y oficinas), los círculos los temas y los
             colores la dimensión del desarrollo abordada. La relación entre dos unidades viene dada por un tema, el tamaño del círculo (tema)
             indica su frecuencia en la agenda de investigación. Los temas más centrales en la red indican unamayor conectividad con divisiones,
             mientras que los temas en la periferia de la red indica temas de nicho, o aborados por una o pocas unidades organizacionales. 
             Para interactuar con la red se puede hacer zoom sobre ella para identificar unidades, temas y relaciones en detalle. 
             De igual manera puede filtarse en la barra izquierda por unidades organizacionales para observar relacions entre dos o más unidades.

             Debajo del primer gráfico se puede realizar un filtro por un tópico determinado y observar las relaciones de ese tópico con las
             unidades que plantean abordarlos en su agenda futura de investigación.
             """)
    
    frecuencia = df_fil_tema
    frecuencia.loc[:,'assigned_topics'] = frecuencia['assigned_topics'].replace('RECURSOS NATURALES', 'REC NATURALES')
    
    frecuencia = frecuencia.merge(clusters, left_on='assigned_topics', right_on="temas", how = "left")
    # Calcular la frecuencia de cada tema y división
    frecuencia_frec = frecuencia.groupby(['assigned_topics', 'Entidad','Cluster']).size().reset_index(name ='frecuencia') # Temas de frecuencia por division

    # Función para crear la visualización de la red
    def create_network_visualization(df):
        # Crear un objeto Network de pyvis
        net = Network(directed=False, height='700px')
        net.toggle_physics(True)

        # Creamos un dic para los colores
        cluster_colors = {
            "1. Desarrollo económico": "#E41B1C",
            "2. Desarrollo social": "#377EB8",
            "3. Sustentabilidad ambiental y gestión de recursos naturales": "#4DAF4A",
            "4. Desarrollo productivo, innovación y aprovechamiento tecnológico": "#984EA3",
            "5. Institucionalidad, gobernanza y temas transversales": "#FF7F00"
        }
        # Vamos a pasar el tamaño de los nodos a una escala logaritmica
        log_sizes = frecuencia_frec["frecuencia"]
        # Normalizamos los tamaños en un rango
        min_size, max_size = log_sizes.min(), log_sizes.max()
        normal_sizes = (log_sizes - min_size)/(max_size - min_size)
        # Escalamos los tamaños
        min_node_size = 30
        max_node_size = 80
        scaled_sizes = normal_sizes *(max_node_size - min_node_size) + min_node_size

        node_sizes = dict(zip(df["assigned_topics"], scaled_sizes))
        node_colors = dict(zip(df["assigned_topics"], df["Cluster"].map(cluster_colors)))
        
        # # Crear conjuntos para almacenar temas y divisiones únicas
        nodos_e = set()
        # Agregar nodos y bordes al grafo
        for index, row in df.iterrows():
            tema = row['assigned_topics']
            division = row['Entidad']
            frecuencia = row['frecuencia']
            
            if tema not in nodos_e:
                net.add_node(tema, color = node_colors.get(tema,"#26619C"), size = node_sizes.get(tema, min_node_size))
                nodos_e.add(tema)
            if division not in nodos_e:
                net.add_node(division, color = "#D99E63", shape= "triangle", size = 40)
                nodos_e.add(division)
            # Agregar nodos de temas y divisiones al grafo
            net.add_edge(tema, division, width = frecuencia/20, color = "#9C2661")    
            # Agregar los nombres de temas y divisiones al conjunto respectivo
      

        options  = """
            var options = {
                "physics": {
                    "forceAtlas2Based": {
                        "springLength": 100
                    },
                    "minVelocity": 0.75,
                    "solver": "forceAtlas2Based"
                }
            }
            """
        net.set_options(options)
        # Guardar la visualización en un archivo HTML temporal
        with tempfile.NamedTemporaryFile(delete= False, suffix= ".html") as temp_file:
            net.save_graph(temp_file.name)
       
        # Mostrar la visualización inicial en Streamlit
        st.components.v1.html(open(temp_file.name, "r").read(), height= 750)
# ============================================================================================
# Legenda
# ============================================================================================

        legend_colors = [
            {"color": "#E41B1C", "label": "1. Desarrollo económico"},
            {"color": "#377EB8", "label": "2. Desarrollo social"},
            {"color": "#4DAF4A", "label": "3. Sustentabilidad ambiental y gestión de recursos naturales"},
            {"color": "#984EA3", "label": "4. Desarrollo productivo, innovación y aprovechamiento tecnológico"},
            {"color": "#FF7F00", "label": "5. Institucionalidad, gobernanza y temas transversales"}
        ]
        with st.container():
            cols = st.columns(len(legend_colors))
            for i, item in enumerate(legend_colors):
                color_box = f'<span style="background-color:{item["color"]};width:20px;height:20px;display:inline-block;margin-right:5px;"></span>'
                cols[i].markdown(f'{color_box} {item["label"]}', unsafe_allow_html=True)
        
        # Crear menú de selección en Streamlit
# ============================================================================================
# Segundo gráfico
# ============================================================================================
        selected_option = st.multiselect(label = "Selecciona un tema de la lista desplegable para ver sus relaciones: ", options= sorted(df["assigned_topics"].unique()))

        # Actualizar la visualización cuando el usuario seleccione una opción
        if selected_option:
            # Crear un nuevo objeto Network solo con los nodos y bordes relacionados con la opción seleccionada
            selected_net = Network(height='500px')
            selected_net.toggle_physics(True)
            selected_nodes = set()

            for index, row in df.iterrows():
                tema = row['assigned_topics']
                division = row['Entidad']
                if tema in selected_option or division in selected_option:
                    color = node_colors.get(tema, "#26619C")
                    if tema not in selected_nodes:
                        selected_net.add_node(tema, color= color, size=node_sizes.get(tema, min_node_size))
                        selected_nodes.add(tema)
                    if division not in selected_nodes:
                        selected_net.add_node(division, color = "#D99E63")
                        selected_nodes.add(division)
                    selected_net.add_edge(tema, division, width = row["frecuencia"]/20, color = "#9C2661")

            # Guardar la visualización actualizada en un archivo HTML temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file_selected:
                        selected_net.save_graph(temp_file_selected.name)            
            # Mostrar la visualización actualizada en Streamlit
            st.components.v1.html(open(temp_file_selected.name, "r").read(), height=600)


    create_network_visualization(frecuencia_frec)


with brechas:
     st.title("Brechas")
     st.write(
    """Relación de la agenda de investigación con las brechas del décalogo. El gráfico agrupa el listado de 
    investigaciones propuestas por brechas identificadas. Luego del gráfico puede observarse el detalle de las 
    investigaciones que indentifican relación brechas y finalmente puede obtenerse un detalle respecto a una brecha específica.""")
     
     df_brechas = brechas_exp[mask]
     brechas_div = df_brechas.groupby(['brechas']).size().reset_index(name = "frecuencia")



     # ordenamos los valores de mayor a menor
     brechas_div.sort_values(by='brechas', inplace=True)
     
     plot = px.bar(brechas_div,
                  title = "Líneas de investigación por brecha atendida",
                  y = "brechas",
                  x = "frecuencia",
                  orientation= "h",
                  text_auto= '0',
                  height= 600, 
                  labels= {"Entidad":"División",
                            "ids":"Total de investigaciones"}
                            )
      #plot.update_layout(xaxis_tickformat = ".0")
     plot.update_traces(textfont_size = 12)
     plot.update_layout(
         #xaxis=dict(dtick=1),
         yaxis=dict(dtick=1))
     st.plotly_chart(plot, use_container_width= True)

     st.write("""
              Brechas:
1. Crecimiento económico bajo, volátil, excluyente y no sostenible con baja creación de empleo formal
2. Elevada desigualdad y baja movilidad y cohesión social
3. Brechas de protección social
4. Sistemas educativos y formación profesional débiles
5. Alta desigualdad de género
6. Desarrollo ambientalmente no sostenible y cambio climático
7. Brecha digital
8. Flujos migratorios intrarregionales crecientes en cantidad y diversidad
9. Insuficiente integración económica regional
10. Espacios fiscales limitados y altos costos de financiamiento
""")


     numero_resultados = brechas_filt.shape[0] # number of availables rows

     st.subheader('Detalle de investigaciones')

     st.markdown(f'*Resultados Disponibles: {numero_resultados}*') 
                                 
   
     
     st.dataframe(
     brechas_filt[['Entidad','Titulo','Detalle','Brechas Atendidas']],
        column_config={
            "Entidad": "Unidad",
            "Titulo": 'Investigación',
            "Detalle": "Pregunta de Investigación"
#            "brechas": "Brechas"        
        },
        hide_index=True,
        use_container_width= True
    )
     
     brecha_df = brechas_div.loc[brechas_div['frecuencia'].idxmax(), 'brechas']


          

     st.subheader('Listados de investigaciones por brechas atendidas')


     brecha = df_brechas['brechas'].unique().tolist() # se crea una lista unica de la columna Division

     brechas_selector = st.selectbox('Seleccionar brecha:', brecha, index=brecha.index(brecha_df))


     # Crea un contenedor vacío para el dataframe
     df_contain = st.empty()

    

     # Dentro del contenedor, crea el dataframe
     with df_contain.container():
        # Código para crear el dataframe filtrado por tema_selector
        df_filtrado_brecha = df_brechas[df_brechas['brechas']==brechas_selector][['Entidad','Titulo','Detalle','assigned_topics']]
        st.write("Investigaciones: ", len(df_filtrado_brecha))
        st.dataframe(df_filtrado_brecha)

    # Actualiza el dataframe dentro del contenedor cuando el tema_selector cambia

     with df_contain.container():
        df_filtrado_tema = df_brechas[df_brechas['brechas']==brechas_selector][['Entidad','Titulo','Detalle','assigned_topics']]
        st.write("Invesigaciones: ", len(df_filtrado_brecha))  
        st.dataframe(df_filtrado_brecha,
                    column_config={
      #      "dc.title": "Título",
      #      "dc.identifier.uri": st.column_config.LinkColumn("Enlace"),
      #      "dc.year": "Año",
      #      "division": "División"    
        },
        hide_index=True,
        use_container_width= True)


with sectores:
     st.title("Sectores impulsores")
     st.write(
    """Relación de la agenda de invesigación futura con los sectores impulsores. El gráfico agrupa el listado de investigaciones
    propuestas por sector impulsor identificado. Luego del gráfico puede observarse el detalle de las investigaciones
    que indentifican relación con sectores impulsores y finalmente puede obtenerse un detalle respecto a un sector específico.
""")
     
     df_sectores = sectores_exp[mask]
     sectores_div = df_sectores.groupby(['sectores']).size().reset_index(name = "frecuencia")

     # ordenamos los valores de mayor a menor
     sectores_div.sort_values(by='sectores', inplace=True)
     
     plot = px.bar(sectores_div,
                  title = "Línea de investigación por sectores impulsores",
                  y = "sectores",
                  x = "frecuencia",
                  orientation= "h",
                  text_auto= '0',
                  height= 600, 
                  labels= {"Entidad":"División",
                            "ids":"Total de investigaciones"}
                            )
      #plot.update_layout(xaxis_tickformat = ".0")
     plot.update_traces(textfont_size = 12)
     plot.update_layout(
         #xaxis=dict(dtick=1),
         yaxis=dict(dtick=1))
     st.plotly_chart(plot, use_container_width= True)

     st.write(""" Sectores:
1. Industria farmacéutica y de ciencias de la vida
2. Industria de dispositivos médicos
3. Manufactura avanzada
4. Exportación de servicios modernos habilitados por las TIC
5. Sociedad del cuidado
6. Servicios intensivos en trabajo
7. Gobierno digital
8. Transición energética 
9. Bioeconomía
10. Electromovilidad
11. Economía circular
12. Agricultura para la seguridad alimentaria
13. Gestión sostenible del agua
14. Turismo sostenible
15. Reubicación geográfica de la producción y de las cadenas globales de valor
    """)
     
     st.markdown('## Detalle de investigaciones')

     numero_resultados = sectores_filt.shape[0] # number of availables rows

     st.markdown(f'*Resultados Disponibles: {numero_resultados}*') 


                                 
   
     
     st.dataframe(
     sectores_filt[['Entidad','Titulo','Detalle','Sectores Atendidos']],
        column_config={
            "Entidad": "Unidad",
            "Titulo": 'Investigación',
            "Detalle": "Pregunta de Investigación"
#            "brechas": "Brechas"        
        },
        hide_index=True,
        use_container_width= True
    )
     
     # sector_df = sectores_div.loc[sectores_div['frecuencia'].idxmax(), 'sectores']


          

     st.subheader('Listados de investigaciones por sectores atendidos')


     sector = df_sectores['sectores'].unique().tolist() # se crea una lista unica de la columna Division

     sectores_selector = st.selectbox('Seleccionar brecha:', sector, index=0) # sector.index(sector_df))


     # Crea un contenedor vacío para el dataframe
     df_contain = st.empty()

    

     # Dentro del contenedor, crea el dataframe
     with df_contain.container():
        # Código para crear el dataframe filtrado por tema_selector
        df_filtrado_sectores = df_sectores[df_sectores['sectores']==sectores_selector][['Entidad','Titulo','Detalle','assigned_topics']]
        st.write("Investigaciones: ", len(df_filtrado_sectores))
        st.dataframe(df_filtrado_sectores)

    # Actualiza el dataframe dentro del contenedor cuando el tema_selector cambia

     with df_contain.container():
        df_filtrado_sectores = df_sectores[df_sectores['sectores']==sectores_selector][['Entidad','Titulo','Detalle','assigned_topics']]
        st.write("Investigaciones: ", len(df_filtrado_sectores))  
        st.dataframe(df_filtrado_sectores,
                    column_config={
      #      "dc.title": "Título",
      #      "dc.identifier.uri": st.column_config.LinkColumn("Enlace"),
      #      "dc.year": "Año",
      #      "division": "División"    
        },
        hide_index=True,
        use_container_width= True)

with chat:
    import streamlit as st
    import pandas as pd
    from langchain_community.chat_models import ChatOpenAI
    from langchain.agents import create_pandas_dataframe_agent
    from langchain.agents.agent_types import AgentType
    from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
    
    # Configuración de la página
    st.set_page_config(page_title="Chat con Base de Datos", page_icon="🤖")
    
    # Cargar el archivo CSV
    uploaded_file = 'datos_full.csv'  # Reemplaza con tu archivo CSV
    
    if not uploaded_file:
        st.warning(
            "This app uses LangChain's `PythonAstREPLTool` which is vulnerable to arbitrary code execution. Please use caution in deploying and sharing this app."
        )
    
    # Cargar los datos
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    
        # Convertir columnas problemáticas a tipos compatibles
        df["Brechas Atendidas"] = df["Brechas Atendidas"].astype(str)
        df["Sectores Atendidos"] = df["Sectores Atendidos"].astype(str)
    
        # Interfaz de usuario
        st.title("Chat con bases de datos agenda de investigación")
        st.write("""Este chatbot permite interactuar con la base de datos de agenda de investigación. Esta contiene los siguientes datos:
    
    1. Entidad o Division
    2. Título de Investigación
    3. Período
    4. Detalle de Investigación
    5. Palabras clave
    6. Decálogo
    7. Gobernanza y capacidades TOPP
    8. Sectores impulsores
    9. Brechas Atendidas
    10. Sectores Atendidos
    11. Preguntas Agenda Común
    12. Tópicos CEPAL
        """)
    
        st.write("""Se recomienda hacer preguntas haciendo mención al o los campos de consulta de manera literal para su mejor funcionamiento
                 (las divisiones están en siglas).
                Ejemplos:
                 
    - Lístame las divisiones            
    - Lístame los títulos de las investigaciones de CELADE
    - Lístame los títulos de las investigaciones relacionadas con cambio climático y su entidad autora
    - Cuáles son los 10 principales tópicos de las investigaciones
    - Cuáles son las brecha atendidas más abordadas por las investigaciones
                """)

    # Clave API de OpenAI
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    openai_api_key = OPENAI_API_KEY

    # Limpiar la conversación
    if "messages" not in st.session_state or st.button("Limpiar conversación"):
        st.session_state["messages"] = [{"role": "assistant", "content": "Cómo puedo ayudarte?"}]

    # Mostrar mensajes anteriores
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Entrada de chat
    if query := st.chat_input(placeholder="De que tratan estos datos?"):
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)

        if not openai_api_key:
            st.info("Por favor, añade tu clave API de OpenAI para continuar.")
            st.stop()

        # Inicializar el modelo de OpenAI
        llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo",
            openai_api_key=openai_api_key,
            streaming=True,
        )

        # Crear el agente
        try:
            pandas_df_agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                handle_parsing_errors=True,
            )
        except Exception as e:
            st.error(f"Error al crear el agente: {e}")
            st.stop()

        # Generar la respuesta
        try:
            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = pandas_df_agent.run(query, callbacks=[st_cb])
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
        except Exception as e:
            st.error(f"Error al generar la respuesta: {e}")

design.footer()
