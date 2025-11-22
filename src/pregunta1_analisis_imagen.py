# Pregunta 1 - Análisis de imagen veterinaria con Hugging Face
# Autor: Andres Gallardo

from transformers import pipeline
from PIL import Image
import json

# 1. NOMBRE DEL ARCHIVO DE LA IMAGEN -------------------------
# AQUI CAMBIA "paciente1.jpg" POR EL NOMBRE DE TU IMAGEN
RUTA_IMAGEN = "paciente1.jpg"

# 2. ABRIR LA IMAGEN -----------------------------------------
imagen = Image.open(RUTA_IMAGEN).convert("RGB")

# 3. MODELO PARA DESCRIPCION DE IMAGEN -----------------------
# (Captioning: describe la imagen en texto)
modelo_descripcion = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base"
)

# 4. MODELO PARA CLASIFICACIÓN (ESPECIE APROXIMADA)
# Modelo simple que SÍ detecta perros, gatos, aves, etc.
modelo_clasificacion = pipeline(
    "image-classification",
    model="microsoft/resnet-50"
)

# 5. MODELO PARA TEXTO EN IMAGEN (OCR SIMPLE) ----------------
# Si la imagen tiene texto impreso (ej: carnet, receta)
modelo_ocr = pipeline(
    "image-to-text",
    model="microsoft/trocr-base-printed"
)

# 6. OBTENER DESCRIPCION -------------------------------------
descripcion_result = modelo_descripcion(imagen)[0]["generated_text"]

# 7. OBTENER CLASIFICACION (ESPECIE / TIPO) ------------------
clasificacion_result = modelo_clasificacion(imagen)[0]
etiqueta_clase = clasificacion_result["label"]
confianza_clase = float(clasificacion_result["score"])

# 8. INTENTAR LEER TEXTO EN LA IMAGEN ------------------------
try:
    ocr_result = modelo_ocr(imagen)[0]["generated_text"]
except Exception:
    ocr_result = "No se detecta texto en la imagen."

# 9. CALCULAR COLORES DOMINANTES (OPCIONAL) ------------------
# Reducimos la imagen para simplificar
imagen_pequena = imagen.resize((50, 50))
# Obtenemos los colores (máx. 2500)
colores = imagen_pequena.getcolors(2500)

# Ordenamos por frecuencia y tomamos los 3 más frecuentes
colores_ordenados = sorted(colores, key=lambda x: x[0], reverse=True)
top_colores = [c[1] for c in colores_ordenados[:3]]  # (R, G, B)

# 10. ARMAR UN DICCIONARIO CON LOS RESULTADOS ----------------
resultado = {
    "especie_o_clase_aproximada": etiqueta_clase,
    "confianza_clasificacion": round(confianza_clase, 4),
    "descripcion_imagen": descripcion_result,
    "texto_detectado": ocr_result,
    "colores_dominantes_rgb": top_colores
}

# 11. IMPRIMIR RESULTADOS EN FORMATO JSON --------------------
print(json.dumps(resultado, indent=4, ensure_ascii=False))
