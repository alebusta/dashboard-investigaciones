# Codigo para definir algunos elementos que se repiten en las páginas
import streamlit as st

# Carga el archivo CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def header():
    c1, c2, c3 = st.columns(3)
    with c3:
        st.image('cepal.png', width=100)


def footer():
    with st.sidebar:
        st.markdown(
        """
        <div class="footer">
            <p>Mapeo Estratégico de conocimiento CEPAL - Julio 2024</p>
        </div>
        """,
        unsafe_allow_html=True
    )