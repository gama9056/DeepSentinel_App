import numpy as np
import rasterio
import os
from super_image import EdsrModel, ImageLoader
from PIL import Image

def mascara_nubes_s2(ds):
    """Identifica nubes y sombras usando la banda SCL."""
    scl = ds.scl
    mask = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)
    return ds.where(~mask)

def obtener_mejor_escena_local(ds, umbral_maximo):
    """Lógica mejorada: selecciona la imagen con menos nubes dentro del AOI."""
    scl = ds.scl
    # 3=Sombra, 8,9,10=Nubes
    nubes_sombras = (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)
    
    # Calcular % de nubes por cada fecha en el set de datos
    porcentaje_nubes = (nubes_sombras.mean(dim=['x', 'y']) * 100).compute()
    
    # Filtrar solo las que cumplen tu criterio
    validas = porcentaje_nubes.where(porcentaje_nubes <= umbral_maximo, drop=True)
    
    if len(validas) == 0:
        return None, None
    
    # Seleccionar la que tiene el MÍNIMO de nubes (la más limpia)
    mejor_idx = validas.argmin().item()
    mejor_porcentaje = validas[mejor_idx].values
    mejor_escena = ds.isel(time=mejor_idx)
    
    return mejor_escena, mejor_porcentaje

def aplicar_ia_edsr(img_np):
    """Transforma la imagen de 10m a 2.5m."""
    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)
    img_pil = Image.fromarray(img_np)
    inputs = ImageLoader.load_image(img_pil)
    preds = model(inputs)
    
    if len(preds.shape) == 4: preds = preds[0]
    img_ia = preds.permute(1, 2, 0).cpu().detach().numpy()
    return (np.clip(img_ia, 0, 1) * 255).astype(np.uint8)
