# Clasificador de reseñas de McDonald´s

Repositorio para un **clasificación de reseñas de McDonald`s en Barcelona** utilizando Machine Learning y NLP

El objetivo es analizar las reseñas de clientes y predecir si son positivas o negativas, mediante técnicas de procesamiento de lenguaje natural y regresión. 

## Estructura del proyecto

reviews_classificator/
│
├── data/ # Carpeta con datasets (ignorados por Git)
├── notebooks/ # Jupyter notebooks de análisis exploratorio
├── utils/ # Scripts de limpieza y preprocesamiento
├── main.ipynb # Script principal para entrenar y evaluar
├── requirements.txt # Dependencias del proyecto
├── README.md # Este archivo
└── .gitignore # Archivos ignorados (data/, venv/, etc.)

## 🔹 Funcionalidades principales

1. Limpieza y preprocesamiento del dataset de reseñas.  
2. Tokenización y análisis de palabras positivas/negativas usando **NLTK Opinion Lexicon**.  
3. Vectorización de texto con **TF-IDF**.  
4. Clasificación de reseñas en **positivo/negativo/neutral** usando **Logistic Regression**.  
5. Métricas de evaluación: accuracy, precision, recall, f1-score.  

---

## 🔹 Requisitos y entorno

Se recomienda **crear un entorno virtual** para instalar las dependencias:

```bash
python -m venv .venv
# Activar entorno
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
# Instalar dependencias
pip install -r requirements.txt
