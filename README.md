# IMDB Sentiment Classification with Deep Learning

> Proyecto Final - Data Science III | Coderhouse
> 
> Análisis profundo de reseñas de películas IMDB usando NLP y Deep Learning para identificar películas verdaderamente excelentes basadas en sentimiento genuino

---

## 📋 Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Problema y Oportunidad](#problema-y-oportunidad)
- [Solución Propuesta](#solución-propuesta)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Pipeline del Proyecto](#pipeline-del-proyecto)
- [Resultados](#resultados)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Cómo Ejecutar](#cómo-ejecutar)
- [Conclusiones y Mejoras Futuras](#conclusiones-y-mejoras-futuras)

---

## 🎯 Descripción General

Este proyecto implementa un **Sistema Inteligente de Clasificación de Sentimientos** que analiza el contenido emocional de reseñas de películas en IMDB. A diferencia de los ratings numéricos tradicionales, nuestro modelo captura el sentimiento genuino del lenguaje natural, permitiendo:

- Identificar películas verdaderamente excelentes más allá del promedio numérico
- Segmentar la calidad de películas según análisis de sentimiento profundo
- Mejorar recomendaciones al entender matices en las opiniones de usuarios
- Aumentar engagement presentando contenido que realmente resonará con cada perfil

**Resultados alcanzados: 85.78% de precisión en clasificación de sentimientos**

---

## ⚠️ Problema y Oportunidad

### El Desafío Actual

IMDB posee aproximadamente 250 millones de reseñas de películas. Aunque cada una incluye una calificación numérica (1-10), estas puntuaciones no cuentan la historia completa:

- **Rating inflation**: Películas populares tienden a tener puntuaciones más altas, independientemente de la calidad real
- **Reseñas contradictorias**: Un usuario puede dar 8/10 pero escribir "excelente" mientras otro da 8/10 escribiendo "decepciona"
- **Información perdida**: El contexto emocional de cada reseña no se captura en un simple número
- **Oportunidad de diferenciación**: Netflix, Disney+ y otros competidores están invirtiendo en IA para personalización

### La Oportunidad Comercial

Si IMDB puede clasificar automáticamente el sentimiento genuino de las reseñas, podrá:

- Crear rankings más inteligentes: "Películas que la gente REALMENTE ama" vs "películas populares"
- Personalizar recomendaciones: Presentar películas basadas en el sentimiento de quién tiene gustos similares
- Monetizar mejor: Ofrecer a productores insights sobre la percepción genuina de sus películas
- Mejorar retención: Usuarios encuentran películas que les encantarán

**Impacto esperado:** Aumento de 15-25% en engagement de usuarios

---

## 💡 Solución Propuesta

### Visión General del Sistema

El sistema consta de tres componentes integrados:

**1. Motor de Procesamiento de Lenguaje Natural (NLP)**
- Limpieza y normalización de reseñas
- Extracción de características semánticas mediante TF-IDF
- Identificación de palabras clave que indican sentimiento

**2. Red Neuronal Profunda (Deep Learning)**
- Clasificación binaria: Reseña Positiva vs Negativa
- Entrenada con 50,000 reseñas de películas reales
- Capaz de capturar matices emocionales complejos

**3. Motor de Recomendación Mejorado**
- Análisis de sentimiento en tiempo real
- Agregación de sentimientos por película
- Ranking inteligente basado en "qué tan genuinamente amada es una película"

---

## 🛠️ Tecnologías Utilizadas

**Librerías Principales:**
- PyTorch: Red neuronal y Deep Learning
- scikit-learn: TF-IDF y métricas de evaluación
- NLTK: Tokenización y lematización
- spaCy: Procesamiento avanzado de lenguaje
- Pandas & NumPy: Manipulación de datos
- Matplotlib & Seaborn: Visualización

**Herramientas de Desarrollo:**
- Google Colab: Entorno de desarrollo
- Hugging Face Datasets: Dataset IMDB
- Git & GitHub: Control de versiones

---

## 🔄 Pipeline del Proyecto

### Fase 1: Exploración y Preprocesamiento

```
Dataset Crudo (50,000 reseñas)
    ↓
Exploratory Data Analysis (EDA)
    • Distribución de clases: 50% Positivas / 50% Negativas
    • Análisis de palabras frecuentes (Top 20)
    • Análisis diferenciado por clase
    ↓
Limpieza con Regex
    • Eliminar URLs, HTML tags, menciones
    • Convertir a minúsculas
    • Remover caracteres especiales
    ↓
Tokenización y Lematización (NLTK vs spaCy)
    • NLTK: 121.24 tokens promedio
    • spaCy: 103.78 tokens promedio
    • Decisión: Usar NLTK (menos agresivo)
    ↓
Vectorización con TF-IDF
    • 5,000 características extraídas
    • Matriz final: 50,000 × 5,000
    • Prioriza palabras discriminantes
```

### Fase 2: Modelado y Entrenamiento

```
Vectores TF-IDF (50,000 × 5,000)
    ↓
TextClassifier(nn.Module)
    • Input: 5,000 features
    • Hidden Layer 1: 256 neuronas + BatchNorm + Dropout(0.5)
    • Hidden Layer 2: 128 neuronas + BatchNorm + Dropout(0.5)
    • Output: 2 clases (Positivo/Negativo)
    • Total parámetros: 1,314,178
    ↓
Entrenamiento
    • Optimizador: Adam (lr=0.001)
    • Loss: CrossEntropyLoss
    • Early Stopping: patience=5
    • Epochs completados: 30
    ↓
Evaluación
    • Accuracy: 85.78%
    • Matriz de confusión generada
    • Reporte de clasificación por clase
```

---

## 📊 Resultados

### Métricas de Desempeño

**Exactitud y Errores:**
- Exactitud General: 85.78% - De 25,000 reseñas de prueba, 21,444 fueron clasificadas correctamente
- Tasa de Error: 14.22% - 3,556 clasificaciones incorrectas de 25,000 total
- Predicciones Correctas: 21,444 reseñas
- Predicciones Incorrectas: 3,556 reseñas

**Desempeño por Clase:**
- Verdaderos Positivos: 10,781 reseñas positivas identificadas correctamente
- Verdaderos Negativos: 10,663 reseñas negativas identificadas correctamente
- Falsos Positivos: 1,837 reseñas negativas clasificadas como positivas
- Falsos Negativos: 1,719 reseñas positivas clasificadas como negativas

**Precisión y Recall por Clase:**
- Precisión Clase Negativa: 85.36% - Calidad de predicciones negativas
- Precisión Clase Positiva: 85.48% - Calidad de predicciones positivas
- Recall Clase Negativa: 85.30% - Cobertura de reseñas negativas
- Recall Clase Positiva: 86.25% - Cobertura de reseñas positivas

### Evolución del Entrenamiento

- Convergencia rápida: Alcanzó máximo desempeño en ~15 épocas
- Sin overfitting: Loss de training y validación convergieron juntos
- Estabilidad: Mejora continua sin fluctuaciones
- Early Stopping: No se activó (modelo mejoró permanentemente)

### Matriz de Confusión

```
                Real Negativo    Real Positivo
Predicho Neg        10,663           1,719
Predicho Pos         1,837          10,781

Distribución:
- Aciertos: 21,444 (85.78%)
- Errores: 3,556 (14.22%)
```

---


## 🚀 Cómo Ejecutar

### Requisitos Previos

- Python 3.8+
- pip o conda

### Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/TuUsuario/sentiment-analysis-imdb.git
cd sentiment-analysis-imdb
```

2. Crear entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Descargar modelos de NLP:
```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords wordnet
```

### Ejecución

**Opción 1: Google Colab (Recomendado)**
- Abrir `notebook/Trabajo_Final_Data_science_III.ipynb` en Google Colab
- Ejecutar celdas secuencialmente
- Los datos se descargan automáticamente de Hugging Face

**Opción 2: Jupyter Local**
```bash
jupyter notebook notebook/Trabajo_Final_Data_science_III.ipynb
```

**Opción 3: Scripts Python**
```bash
python src/preprocessing.py
python src/training.py
python src/evaluation.py
```

---

## 🔍 Análisis Técnico Detallado

### 1. Preprocesamiento

**Limpieza con Regex:**
- Eliminación de URLs, etiquetas HTML, menciones
- Normalización a minúsculas
- Remoción de caracteres especiales

**Tokenización y Lematización:**
- Comparación NLTK vs spaCy
- NLTK elegido por ser menos agresivo (mantiene más información)
- Promedio final: ~120 tokens por reseña

### 2. Vectorización TF-IDF

**¿Por qué TF-IDF y no conteo simple?**

- TF (Term Frequency): Cuenta frecuencia de palabras en cada documento
- IDF (Inverse Document Frequency): Penaliza palabras comunes en el corpus
- Resultado: Prioriza palabras discriminantes ("excellent", "terrible") y penaliza comunes ("movie", "film")

**Comparación:**
```
Conteo Simple:    [excellent:3, movie:47, great:5, film:32]
TF-IDF:           [excellent:0.85, movie:0.05, great:0.72, film:0.08]
                   → Detecta automáticamente palabras importantes
```

### 3. Arquitectura del Modelo

```
Input (5,000 features TF-IDF)
    ↓
Dense Layer 1 (256 neurons)
    ↓ BatchNorm1d(256)
    ↓ ReLU Activation
    ↓ Dropout(0.5)
Dense Layer 2 (128 neurons)
    ↓ BatchNorm1d(128)
    ↓ ReLU Activation
    ↓ Dropout(0.5)
Dense Layer 3 (2 neurons - Output)
    ↓
Softmax → Probabilidades [Negativo, Positivo]
```

**Componentes Implementados:**
- ✓ Clase TextClassifier(nn.Module)
- ✓ 3 capas densas (2 ocultas + 1 salida)
- ✓ Batch Normalization en ambas capas ocultas
- ✓ Dropout p=0.5 para regularización
- ✓ Optimizador Adam (lr=0.001)
- ✓ CrossEntropyLoss
- ✓ Early Stopping (patience=5)

### 4. Entrenamiento

**Configuración:**
- Batch size: 64
- Epochs: 30 (completados exitosamente)
- Learning rate: 0.001
- Loss final: 0.021

**Monitoreo:**
- Loss de entrenamiento descendió de 0.253 a 0.021
- Sin overfitting significativo
- Convergencia suave y estable

---

## 📈 Limitaciones y Consideraciones

### 1. Características del Dataset
- Dataset IMDB es muy limpio (reseñas altamente polarizadas)
- No refleja completamente la dificultad real de textos más ambiguos
- Estructura consistente de las reseñas

### 2. Errores Residuales
Los 3,556 errores (14.22%) se deben principalmente a:
- Reseñas sarcásticas: "This movie was so bad it's actually amazing"
- Sentimientos mezclados: "Great acting, terrible plot"
- Opiniones matizadas: Críticas constructivas con puntuación alta

### 3. Limitaciones de Generalización
El modelo entrenado en IMDB podría tener desempeño inferior en:
- Reseñas de productos (Amazon, eBay) - lenguaje más técnico
- Redes sociales (Twitter, Instagram) - lenguaje coloquial
- Comentarios de noticias - contexto político/ideológico
- Feedback de clientes - más neutral y detallado

### 4. Limitaciones de TF-IDF
- No captura orden ni contexto de palabras
- No entiende sinonimia ("amazing" = "great")
- Dificultades con negaciones ("not good" vs "very good")

---

## 🚀 Mejoras Futuras

### Corto Plazo
- Ajuste de hiperparámetros (learning rates, arquitectura)
- Data Augmentation (sinónimos, paráfrasis)
- Ampliar dataset a 100,000+ registros

### Mediano Plazo
- Word Embeddings (Word2Vec, GloVe, FastText)
- Arquitecturas Secuenciales (LSTM, GRU)
- Análisis de sentimiento por aspecto

### Largo Plazo
- Transformers (BERT, RoBERTa)
- Transfer learning desde modelos preentrenados
- API REST para predicciones en tiempo real
- Dashboard interactivo

---

## 📚 Conclusiones

### Logros Alcanzados

Este proyecto demostró exitosamente:
- Pipeline Completo de NLP: Desde texto crudo a predicciones
- Preprocesamiento Efectivo: Limpieza, tokenización, lematización
- Vectorización Inteligente: TF-IDF capturando semántica
- Deep Learning Funcional: Arquitectura bien diseñada
- Resultados Validados: 85.78% de accuracy en datos nuevos

### Aprendizajes Clave

- TF-IDF sigue siendo poderoso: Extremadamente efectivo para clasificación de texto
- El preprocesamiento es crítico: 80% del éxito depende de la calidad de entrada
- Batch Normalization + Dropout: Combinación ganadora para datasets medianos
- Early Stopping protege el modelo: Previene overfitting sin comprometer aprendizaje

### Reflexión Final

El 85.78% de accuracy validó nuestra metodología. Los principios aplicados son escalables a problemas más complejos:
- En producción esperaríamos 80-85% en datos más ruidosos
- Este éxito demuestra que la arquitectura es sólida y generalizable
- El pipeline de NLP+DL es reproducible para otros dominios

---

## 👤 Autor

**Marcos** - Data Scientist en formación
- 🎓 Estudiante de Data Science | Coderhouse
- 💼 +4 años de experiencia en análisis y gestión de datos
- 🔧 Stack técnico: Python, SQL, Power BI, Tableau, PyTorch

---

## 📞 Contacto

Para preguntas, sugerencias o colaboraciones:
- Email: marcospraga.mp@gmail.com
- LinkedIn: [linkedin.com/in/marcos](https://www.linkedin.com/in/marcospraga/)

---

## 📝 Licencia

Este proyecto está bajo la licencia MIT.

---

## 🙏 Agradecimientos

- Coderhouse: Por la excelente formación en Data Science
- Hugging Face: Por proporcionar el dataset IMDB
- Comunidad de PyTorch: Por la documentación y soporte

---
