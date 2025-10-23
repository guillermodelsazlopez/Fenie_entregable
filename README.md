# Fenie_entregable
Feníe Challenge | GEN-AI AL RESCATE!   


<pre> Fenie/
├─ .env
├─ docker-compose.yml
├─ requirements.txt
├─ dataset_prueba.csv
├─ dataset_DistilBert.csv
├─ dataset_mDeBERTa.csv
└─ src/
   ├─ app_streamlit.py
   ├─ classify.py
   ├─ config.py
   ├─ embeddings.py
   ├─ rag_ollama.py
   └─ ingest_qdrant.py
 </pre>

# Instalar dependencias básicas

python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Levantar Qdrant
<pre>docker compose up -d</pre>

# Variables de entorno
Se encuentran en el archivo llamado .env en la raíz del proyecto. 
Al usar Ollama para el RAG no es necesario incluir API KEY

# Instalar y correr Ollama (Servicio para correr LLMs en local)
## Descarga Ollama:
https://ollama.com/download
Una vez descargado:
<pre>ollama pull llama3
ollama run llama3</pre>

Decidí usar llama3 debido a que en proyectos similares he tenido buenos resultados. Ollama facilita una gran variad de modelos para distintos proveedores (mistral, deepseek, google...)

# Clasificación Transformer

Una vez levantada la Base de datos Vectorial y el LLM se procede a la clasifición

En la ruta del documento encontrareís distintos archivos de resultados de modelos diferentes.
La elección de DistilBert fué la primera debido aque hace un tiempo me leí el paper y casi igualaba el rendimiento de Bert corriendo con muchos menos recursos, pese a esto, su rendimiento en castellano es bastante pobre.
Clasificar los correos y crear la colección en Qdrant

<pre>python -m src.classify --input dataset_prueba.csv --output emails_pred.csv --to-qdrant</pre>

Esto vuelca los datos directamente a la vectorial, pero también genera un archivo en el root con el ouput que permite hacer el upload más tarde al front.

# Interfaz

<pre>python -m streamlit run src/app_streamlit.py</pre>

Al levantar el servicio de streamlit nos vamos a:

http://localhost:8501

# Uso de la aplicación

1. Carga tu CSV con los correos clasificados (emails_pred.csv).
2. Filtra por tipo, fecha o nivel de confianza.
3. Se pueden corregir etiquetas si es necesario.
4. En la sección RAG, haz preguntas naturales


