# PetGuard Vision IA – Gallardo Quevedo

Proyecto de análisis veterinario asistido por Inteligencia Artificial utilizando modelos de HuggingFace.  
El sistema permite describir imágenes, detectar la especie aproximada, extraer texto y obtener los colores dominantes de una fotografía veterinaria.

---

## Objetivo del Proyecto
Implementar un análisis automático de imágenes veterinarias mediante modelos de IA preentrenados, logrando:

- Obtener una descripción textual de la imagen.
- Clasificar la especie o clase aproximada.
- Extraer texto presente en la imagen (OCR).
- Obtener colores dominantes en la imagen.
- Integrar todo en un script ejecutable en Python para uso rápido y reproducible.

---

## Estructura del Repositorio
ppetguard-vision-ia-gallardoquevedo/
│
├── assets/
│   └── paciente1.jpg
│
├── src/
│   └── pregunta1_analisis_imagen.py
│
└── README.md
---

## Instrucciones de ejecución

1. Instalar dependencias:
```bash
pip install transformers pillow torch matplotlib
python src/pregunta1_analisis_imagen.py

Revisar en consola la salida en formato JSON con:

	•	descripción de la imagen
	•	clasificación aproximada
	•	texto detectado
	•	colores dominantes

🔧 Dependencias y versiones recomendadas

## 🔧 Dependencias y versiones recomendadas

| Librería       | Versión recomendada |
|----------------|----------------------|
| Python         | 3.10 – 3.13          |
| Transformers   | 4.45+                |
| Pillow (PIL)   | 10+                  |
| PyTorch        | 2.2+                 |

----

Sobre el proyecto
Este repositorio forma parte de una práctica universitaria del curso Inteligencia Artificial Generativa, enfocada en el uso de modelos vision-text modernos para análisis veterinario automatizado.
