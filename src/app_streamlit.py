import io
import pandas as pd
import streamlit as st
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from src.config import SETTINGS
from src.rag_ollama import ask
import datetime


fenie_green = "#009639"  # verde corporativo
fenie_white = "#FFFFFF"
fenie_gray = "#F4F4F4"

st.set_page_config(page_title="Feníe Energía | Clasificación de Emails + RAG", page_icon="⚡", layout="wide")


load_dotenv()

st.title(" Clasificación de emails + RAG (Qdrant)")

# Sidebar: conexión
with st.sidebar:
    st.header("Conexión")
    qdrant_url = st.text_input("Qdrant URL", value=SETTINGS.qdrant_url)
    collection = st.text_input("Colección", value=SETTINGS.qdrant_collection)
    top_k = st.slider("k (RAG)", 1, 20, 5)
    client = QdrantClient(url=qdrant_url)

# Carga CSV revisable
st.subheader("Revisar Datos clasificados")
uploaded = st.file_uploader("Sube el CSV con predicciones (fecha, remitente, texto, etiqueta_predicha, confianza, id)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    # Filtros
    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        tipos = ["(todos)"] + sorted([t for t in df["etiqueta_predicha"].dropna().unique()])
        tipo = st.selectbox("Filtrar por tipo", tipos)
    with col2:
        today = datetime.date.today()
        fecha_rango = st.date_input(
            "Rango de fechas (opcional)",
            value=(datetime.date(2021, 12, 31), datetime.date(2026, 12, 31))
        )
        fecha_min, fecha_max = fecha_rango
    with col3:
        conf_min = st.slider("Confianza mínima", 0.0, 1.0, 0.0, 0.01)

    mask = (df["confianza"] >= conf_min)
    if tipo != "(todos)":
        mask &= df["etiqueta_predicha"] == tipo
    if fecha_min:
        mask &= pd.to_datetime(df["fecha"], errors="coerce") >= pd.to_datetime(fecha_min)
    if fecha_max:
        mask &= pd.to_datetime(df["fecha"], errors="coerce") <= pd.to_datetime(fecha_max)

    st.dataframe(df[mask].reset_index(drop=True), use_container_width=True)

    # Corrección manual de etiquetas
    st.markdown("### Corrección de etiqueta")
    idx = st.number_input("Índice de fila", min_value=0, max_value=len(df)-1, value=0)
    nueva = st.selectbox("Nueva etiqueta", ["Queja", "Petición de servicio", "Sugerencia de mejora"])
    if st.button("Aplicar corrección"):
        df.loc[idx, "etiqueta_predicha"] = nueva
        st.success("Etiqueta actualizada en memoria (no persiste aún)")

    # Exportación
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    st.download_button("💾 Descargar CSV revisado", data=buf.getvalue(), file_name="emails_revisado.csv", mime="text/csv")

st.divider()

# RAG Chat
st.subheader("Pregunta a tus correos (RAG)")
q = st.text_input("Pregunta en lenguaje natural (ej.: '¿cuáles son las quejas más comunes?')")
if st.button("Consultar") and q:
    with st.spinner("Buscando y redactando respuesta..."):
        answer, docs = ask(client, q, top_k=top_k)
    st.markdown("#### Respuesta")
    st.write(answer)
    st.markdown("#### Documentos relevantes")
    st.dataframe(pd.DataFrame(docs), use_container_width=True)