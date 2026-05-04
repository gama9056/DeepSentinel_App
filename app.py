import streamlit as st
import geopandas as gpd
import numpy as np
import rasterio
import tempfile
import torch
from pystac_client import Client
from odc.stac import load
from super_image import EdsrModel, ImageLoader
from PIL import Image
import folium
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURACIÓN INICIAL
# ==========================================
st.set_page_config(page_title="DeepSentinel v2", layout="wide")

st.title("🛰️ DeepSentinel: Monitoreo de Precisión")
st.subheader("Análisis centrado en AOI con Super-Resolución IA")

# ==========================================
# CACHE DEL MODELO IA
# ==========================================
@st.cache_resource
def cargar_modelo():
    # Forzar CPU si no hay CUDA disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)
    model = model.to(device)
    model.eval()
    return model, device

# ==========================================
# FUNCIONES
# ==========================================
def obtener_mejor_escena_local(ds, umbral_maximo):
    if 'scl' not in ds:
        return None, None
    
    scl = ds.scl
    nubes_sombras = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)
    
    try:
        porcentaje_nubes = (nubes_sombras.mean(dim=['x', 'y']) * 100).compute()
        validas = porcentaje_nubes.where(porcentaje_nubes <= umbral_maximo, drop=True)
        
        if len(validas) == 0:
            return None, None
        
        mejor_idx = validas.argmin().item()
        mejor_porcentaje = float(validas[mejor_idx].values)
        mejor_escena = ds.isel(time=mejor_idx)
        
        return mejor_escena, mejor_porcentaje
    except Exception as e:
        st.error(f"Error analizando nubes: {str(e)}")
        return None, None

def aplicar_ia_edsr(img_orig_norm):
    model, device = cargar_modelo()
    
    # Convertir a tensor y mover al dispositivo
    img_pil = Image.fromarray(img_orig_norm)
    
    # Asegurar que la imagen está en RGB
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    
    inputs = ImageLoader.load_image(img_pil)
    inputs = inputs.unsqueeze(0) if inputs.dim() == 3 else inputs
    inputs = inputs.to(device)
    
    with torch.no_grad():
        preds = model(inputs)
    
    if len(preds.shape) == 4:
        preds = preds[0]
    
    # Mover a CPU y convertir a numpy
    img_ia_tensor = preds.permute(1, 2, 0).cpu()
    img_ia_np = img_ia_tensor.numpy()
    
    # Normalizar correctamente
    img_ia_np = (img_ia_np * 255).clip(0, 255).astype(np.uint8)
    
    return img_ia_np

def exportar_geotiff(img, ref_dataset):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        try:
            # Obtener transformada de referencia
            transform = ref_dataset.rio.transform()
            crs = ref_dataset.rio.crs
            
            # Asegurar que la imagen tiene el tamaño correcto
            height, width = img.shape[:2]
            
            with rasterio.open(
                tmp.name,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=3,
                dtype=img.dtype,
                crs=crs,
                transform=transform
            ) as dst:
                for i in range(3):
                    dst.write(img[:, :, i], i + 1)
            
            return tmp.name
        except Exception as e:
            st.error(f"Error exportando GeoTIFF: {str(e)}")
            return None

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("Configuración")
    fecha_inicio = st.date_input("Fecha Inicio", value=None)
    fecha_fin = st.date_input("Fecha Fin", value=None)
    umbral_nubes = st.slider("Máximo de nubes (%)", 0, 100, 10)
    archivo_kml = st.file_uploader("Sube tu AOI (KML/KMZ)", type=['kml', 'kmz'])
    
    # Opción adicional
    usar_ia = st.checkbox("Aplicar Super-Resolución IA", value=True)

# ==========================================
# PREVISUALIZACIÓN AOI
# ==========================================
confirmar = False
gdf = None

if archivo_kml:
    try:
        # Guardar archivo temporal para leer
        with tempfile.NamedTemporaryFile(delete=False, suffix='.kml') as tmp_kml:
            tmp_kml.write(archivo_kml.getvalue())
            tmp_kml_path = tmp_kml.name
        
        gdf = gpd.read_file(tmp_kml_path)
        gdf = gdf.to_crs(epsg=4326)
        
        st.subheader("Vista previa del AOI")
        
        # Calcular centro del AOI
        centro_lat = gdf.geometry.centroid.y.mean()
        centro_lon = gdf.geometry.centroid.x.mean()
        centro = [centro_lat, centro_lon]
        
        mapa = folium.Map(location=centro, zoom_start=12, tiles="OpenStreetMap")
        
        # Añadir el polígono con estilo
        folium.GeoJson(
            gdf,
            style_function=lambda x: {'color': 'red', 'weight': 2, 'fillOpacity': 0.3}
        ).add_to(mapa)
        
        st_folium(mapa, width=700, height=400)
        
        confirmar = st.checkbox("✅ Confirmo que el AOI es correcto")
        
    except Exception as e:
        st.error(f"Error leyendo el archivo KML: {str(e)}")

# ==========================================
# PROCESAMIENTO PRINCIPAL
# ==========================================
if archivo_kml and fecha_inicio and fecha_fin and confirmar and gdf is not None:
    
    if st.button("Buscar y Procesar Imagen"):
        
        with st.spinner("Buscando imágenes..."):
            try:
                bbox = [gdf.total_bounds[0], gdf.total_bounds[1], 
                       gdf.total_bounds[2], gdf.total_bounds[3]]
                
                client = Client.open("https://earth-search.aws.element84.com/v1")
                search = client.search(
                    collections=["sentinel-2-l2a"],
                    bbox=bbox,
                    datetime=f"{fecha_inicio}/{fecha_fin}"
                )
                
                items = list(search.get_items())[:8]
                
                if not items:
                    st.warning("No se encontraron imágenes en el período seleccionado")
                    st.stop()
                
            except Exception as e:
                st.error(f"Error en la búsqueda: {str(e)}")
                st.stop()
        
        with st.spinner("Cargando datos satelitales..."):
            try:
                ds = load(
                    items,
                    bands=["red", "green", "blue", "scl"],
                    bbox=bbox,
                    epsg=4326,
                    resolution=10,
                    groupby='auto'
                )
            except Exception as e:
                st.error(f"Error cargando datos: {str(e)}")
                st.stop()
        
        with st.spinner("Analizando calidad de imágenes..."):
            mejor_escena, porcentaje = obtener_mejor_escena_local(ds, umbral_nubes)
        
        if mejor_escena is not None:
            st.success(f"✅ Imagen óptima encontrada con {porcentaje:.2f}% de nubes")
            
            # Construir imagen RGB
            img_np = np.stack([
                mejor_escena.red.values,
                mejor_escena.green.values,
                mejor_escena.blue.values
            ], axis=-1)
            
            # Limpiar valores NaN o inf
            img_np = np.nan_to_num(img_np, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalización por percentiles
            p2, p98 = np.percentile(img_np, (2, 98))
            
            if p98 - p2 < 1e-6:
                img_norm = np.zeros_like(img_np, dtype=np.uint8)
            else:
                img_norm = np.clip(
                    (img_np - p2) / (p98 - p2) * 255, 0, 255
                ).astype(np.uint8)
            
            # Mostrar resultados
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(img_norm, caption="📷 Sentinel-2 Original (10m)", use_container_width=True)
            
            if usar_ia:
                with col2:
                    with st.spinner("🧠 Aplicando Super-Resolución con IA..."):
                        try:
                            img_ia = aplicar_ia_edsr(img_norm)
                            st.image(img_ia, caption="✨ Super-resolución IA (~2.5m visual)", use_container_width=True)
                        except Exception as e:
                            st.error(f"Error en el modelo IA: {str(e)}")
                            img_ia = None
            else:
                with col2:
                    st.info("Super-resolución IA desactivada")
                    img_ia = None
            
            # Sección de descargas
            st.subheader("📥 Descargas")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                tiff_original = exportar_geotiff(img_norm, mejor_escena)
                if tiff_original:
                    with open(tiff_original, "rb") as f:
                        st.download_button(
                            "⬇️ Descargar Sentinel Original",
                            f,
                            file_name="sentinel_10m.tif",
                            mime="image/tiff"
                        )
            
            if usar_ia and img_ia is not None:
                with col_b:
                    tiff_ia = exportar_geotiff(img_ia, mejor_escena)
                    if tiff_ia:
                        with open(tiff_ia, "rb") as f:
                            st.download_button(
                                "⬇️ Descargar Mejora IA",
                                f,
                                file_name="sentinel_ia.tif",
                                mime="image/tiff"
                            )
        
        else:
            st.warning(f"No se encontró imagen con menos de {umbral_nubes}% de nubes")
else:
    st.info("📋 Configura las fechas, sube un archivo KML/KMZ y confirma el AOI para comenzar")
