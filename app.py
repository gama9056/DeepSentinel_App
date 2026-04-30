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
    return EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)

# ==========================================
# FUNCIONES
# ==========================================
def obtener_mejor_escena_local(ds, umbral_maximo):
    scl = ds.scl
    nubes_sombras = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)

    porcentaje_nubes = (nubes_sombras.mean(dim=['x', 'y']) * 100).compute()
    validas = porcentaje_nubes.where(porcentaje_nubes <= umbral_maximo, drop=True)

    if len(validas) == 0:
        return None, None

    mejor_idx = validas.argmin().item()
    mejor_porcentaje = validas[mejor_idx].values
    mejor_escena = ds.isel(time=mejor_idx)

    return mejor_escena, mejor_porcentaje

def aplicar_ia_edsr(img_orig_norm):
    model = cargar_modelo()
    img_pil = Image.fromarray(img_orig_norm)
    inputs = ImageLoader.load_image(img_pil)
    preds = model(inputs)

    if len(preds.shape) == 4:
        preds = preds[0]

    img_ia_np = preds.permute(1, 2, 0).cpu().detach().numpy()
    return (np.clip(img_ia_np, 0, 1) * 255).astype(np.uint8)

def exportar_geotiff(img, ref_dataset):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        transform = ref_dataset.rio.transform()
        crs = ref_dataset.rio.crs

        with rasterio.open(
            tmp.name,
            'w',
            driver='GTiff',
            height=img.shape[0],
            width=img.shape[1],
            count=3,
            dtype=img.dtype,
            crs=crs,
            transform=transform
        ) as dst:
            for i in range(3):
                dst.write(img[:, :, i], i + 1)

        return tmp.name

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("Configuración")
    fecha_inicio = st.date_input("Fecha Inicio", value=None)
    fecha_fin = st.date_input("Fecha Fin", value=None)
    umbral_nubes = st.slider("Máximo de nubes (%)", 0, 100, 10)
    archivo_kml = st.file_uploader("Sube tu AOI (KML/KMZ)", type=['kml', 'kmz'])

# ==========================================
# PREVISUALIZACIÓN AOI
# ==========================================
confirmar = False

if archivo_kml:
    gdf = gpd.read_file(archivo_kml)
    gdf = gdf.to_crs(epsg=4326)

    st.subheader("Vista previa del AOI")

    centro = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    mapa = folium.Map(location=centro, zoom_start=14, tiles="OpenStreetMap")

    folium.GeoJson(gdf).add_to(mapa)

    st_folium(mapa, width=700, height=400)

    confirmar = st.checkbox("✅ Confirmo que el AOI es correcto")

# ==========================================
# PROCESAMIENTO PRINCIPAL
# ==========================================
if archivo_kml and fecha_inicio and fecha_fin and confirmar:

    if st.button("Buscar y Procesar Imagen"):

        with st.spinner("Buscando imágenes..."):
            gdf = gdf.to_crs(epsg=4326)
            bbox = gdf.total_bounds.tolist()

            client = Client.open("https://earth-search.aws.element84.com/v1")
            search = client.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=f"{fecha_inicio}/{fecha_fin}"
            )

            items = list(search.get_items())[:8]

        if items:

            with st.spinner("Cargando datos satelitales..."):
                ds = load(
                    items,
                    bands=["red", "green", "blue", "scl"],
                    bbox=bbox,
                    epsg=4326,
                    resolution=10
                )

            with st.spinner("Analizando calidad..."):
                mejor_escena, porcentaje = obtener_mejor_escena_local(ds, umbral_nubes)

            if mejor_escena is not None:

                st.success(f"Imagen óptima con {porcentaje:.2f}% de nubes")

                img_np = np.stack(
                    [mejor_escena.red, mejor_escena.green, mejor_escena.blue],
                    axis=-1
                )

                img_np = np.nan_to_num(img_np).astype(float)

                p2, p98 = np.percentile(img_np, (2, 98))

                if p98 - p2 == 0:
                    img_norm = np.zeros_like(img_np, dtype=np.uint8)
                else:
                    img_norm = np.clip(
                        (img_np - p2) / (p98 - p2) * 255, 0, 255
                    ).astype(np.uint8)

                col1, col2 = st.columns(2)

                with col1:
                    st.image(img_norm, caption="Sentinel-2 Original (10m)")

                with col2:
                    with st.spinner("Aplicando IA..."):
                        img_ia = aplicar_ia_edsr(img_norm)
                        st.image(img_ia, caption="Super-resolución IA (~2.5m visual)")

                # EXPORTACIÓN
                st.subheader("Descargas")

                tiff_original = exportar_geotiff(img_norm, mejor_escena)
                tiff_ia = exportar_geotiff(img_ia, mejor_escena)

                with open(tiff_original, "rb") as f:
                    st.download_button(
                        "⬇️ Descargar Sentinel (10m)",
                        f,
                        file_name="sentinel_10m.tif"
                    )

                with open(tiff_ia, "rb") as f:
                    st.download_button(
                        "⬇️ Descargar Mejora IA",
                        f,
                        file_name="sentinel_ia.tif"
                    )

            else:
                st.warning("No se encontró imagen dentro del umbral de nubes")

        else:
            st.error("No se encontraron imágenes")

else:
    st.info("Configura fechas, sube AOI y confirma para iniciar")
