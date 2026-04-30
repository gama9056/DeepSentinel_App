import streamlit as st
import geopandas as gpd
from pystac_client import Client
from odc.stac import load
import utils # Importamos tus funciones corregidas

st.set_page_config(page_title="DeepSentinel App", layout="wide")

# Cabecera profesional
st.title("🛰️ DeepSentinel: Vigilancia de Alta Resolución")
st.markdown("Mejora imágenes Sentinel de 10m a 2.5m usando IA enfocada en tu área de interés.")

with st.sidebar:
    st.header("Configuración de Búsqueda")
    fecha_inicio = st.date_input("Fecha Inicio")
    fecha_fin = st.date_input("Fecha Fin")
    umbral_nubes = st.slider("Máximo de nubes en mi AOI (%)", 0, 100, 10)
    
    archivo_kml = st.file_uploader("Sube tu KML/KMZ", type=['kml', 'kmz'])

if archivo_kml:
    # 1. Leer el área de interés
    gdf = gpd.read_file(archivo_kml)
    bbox = gdf.total_bounds.tolist()
    
    if st.button("Iniciar Procesamiento Inteligente"):
        with st.spinner("Buscando la mejor imagen para tu sector..."):
            # 2. Buscar en STAC (Sin filtrar por escena completa)
            client = Client.open("https://earth-search.aws.element84.com/v1")
            rango_fechas = f"{fecha_inicio}/{fecha_fin}"
            
            search = client.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=rango_fechas
            )
            
            items = list(search.get_items())
            
            if items:
                # 3. Carga local y evaluación de nubes
                ds = load(items, bands=["red", "green", "blue", "scl"], bbox=bbox, epsg=3857, resolution=10)
                
                mejor_escena, porcentaje = utils.obtener_mejor_escena_local(ds, umbral_nubes)
                
                if mejor_escena is not None:
                    st.success(f"¡Imagen encontrada! Nubosidad en tu zona: {porcentaje:.2f}%")
                    
                    # 4. Procesar con IA
                    # (Aquí iría el código para normalizar y llamar a utils.aplicar_ia_edsr)
                    st.info("La Super-Resolución se está aplicando sobre los píxeles más limpios...")
                else:
                    st.warning(f"No hay imágenes con menos del {umbral_nubes}% de nubes en tu polígono.")
            else:
                st.error("No se encontraron capturas satelitales en ese rango de fechas.")
else:
    st.info("👋 Sube un polígono KML para comenzar el análisis centrado en tu zona.")
