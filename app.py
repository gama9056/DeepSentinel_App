import streamlit as st
import geopandas as gpd
import numpy as np
import rasterio
import os
import torch
from pystac_client import Client
from odc.stac import load
from super_image import EdsrModel, ImageLoader
from PIL import Image

# ==========================================
# FUNCIONES LÓGICAS (Antes en utils.py)
# ==========================================

def obtener_mejor_escena_local(ds, umbral_maximo):
    """Selecciona la imagen con menos nubes dentro del AOI del usuario."""
    scl = ds.scl
    # Clases SCL: 3=Sombra, 8=Nube Media, 9=Nube Alta, 10=Cirrus
    nubes_sombras = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)
    
    # Calcular % de nubes promediando solo el área del recorte
    porcentaje_nubes = (nubes_sombras.mean(dim=['x', 'y']) * 100).compute()
    
    # Filtrar por el umbral indicado por el usuario
    validas = porcentaje_nubes.where(porcentaje_nubes <= umbral_maximo, drop=True)
    
    if len(validas) == 0:
        return None, None
    
    # Priorizar la excelencia: Elegimos la que tenga el valor MÍNIMO real[cite: 1]
    mejor_idx = validas.argmin().item()
    mejor_porcentaje = validas[mejor_idx].values
    mejor_escena = ds.isel(time=mejor_idx)
    
    return mejor_escena, mejor_porcentaje

def aplicar_ia_edsr(img_orig_norm):
    """Transforma el mosaico de 10m a 2.5m usando el modelo EDSR."""
    # Cargamos el modelo pre-entrenado (Escala 4x)[cite: 1]
    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)
    img_pil = Image.fromarray(img_orig_norm)
    inputs = ImageLoader.load_image(img_pil)
    preds = model(inputs)
    
    if len(preds.shape) == 4: preds = preds[0]
    img_ia_np = preds.permute(1, 2, 0).cpu().detach().numpy()
    return (np.clip(img_ia_np, 0, 1) * 255).astype(np.uint8)

# ==========================================
# INTERFAZ DE USUARIO (Streamlit)
# ==========================================

st.set_page_config(page_title="DeepSentinel v2", layout="wide")

st.title("🛰️ DeepSentinel: Monitoreo de Precisión")
st.subheader("Análisis centrado en AOI con Super-Resolución IA")

with st.sidebar:
    st.header("Configuración")
    fecha_inicio = st.date_input("Fecha Inicio", value=None)
    fecha_fin = st.date_input("Fecha Fin", value=None)
    umbral_nubes = st.slider("Máximo de nubes aceptable en mi zona (%)", 0, 100, 10)
    
    archivo_kml = st.file_uploader("Sube tu polígono (KML o KMZ)", type=['kml', 'kmz'])

if archivo_kml and fecha_inicio and fecha_fin:
    # Leer KML y definir el área de búsqueda exacta[cite: 1]
    gdf = gpd.read_file(archivo_kml)
    bbox = gdf.total_bounds.tolist()
    
    if st.button("Buscar y Mejorar Imagen"):
        with st.spinner("Analizando calidad local de cada captura..."):
            client = Client.open("https://earth-search.aws.element84.com/v1")
            search = client.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=f"{fecha_inicio}/{fecha_fin}"
            )
            
            items = list(search.get_items())
            
            if items:
                # Cargamos solo los datos del área recortada[cite: 1]
                ds = load(items, bands=["red", "green", "blue", "scl"], bbox=bbox, epsg=3857, resolution=10)
                
                # Ejecutar tu lógica de "La mejor de todas"
                mejor_escena, porcentaje = obtener_mejor_escena_local(ds, umbral_nubes)
                
                if mejor_escena is not None:
                    st.success(f"Imagen óptima detectada con {porcentaje:.2f}% de nubes locales.")
                    
                    # Normalización básica para visualización
                    img_np = np.stack([mejor_escena.red, mejor_escena.green, mejor_escena.blue], axis=-1)
                    img_np = np.nan_to_num(img_np).astype(float)
                    p2, p98 = np.percentile(img_np, (2, 98))
                    img_norm = np.clip((img_np - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(img_norm, caption="Original Sentinel (10m)")
                    
                    with col2:
                        with st.spinner("IA aplicando Super-Resolución (2.5m)..."):
                            img_ia = aplicar_ia_edsr(img_norm)
                            st.image(img_ia, caption="Mejorado con IA (~2.5m)")
                else:
                    st.warning(f"Ninguna imagen bajó del {umbral_nubes}% en tu sector. Intenta ampliar el rango de fechas.")
            else:
                st.error("No se encontraron datos para esas fechas.")
else:
    st.info("Configura las fechas y sube tu KML para iniciar la vigilancia.")
