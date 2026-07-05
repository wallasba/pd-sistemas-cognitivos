# ui/analysis.py
import streamlit as st
import matplotlib.pyplot as plt
from src.analyzer import get_stats, generate_wordcloud, plot_distributions

def render_analysis(df):
    """Renderiza estatísticas, distribuições e nuvem de palavras."""
    st.subheader("📊 Estatísticas Descritivas")
    stats = get_stats(df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de artigos", stats['total'])
    col2.metric("Fontes distintas", len(stats['sources']))
    col3.metric("Período", f"{min(stats['years'])} - {max(stats['years'])}")
    
    # Distribuições
    fig = plot_distributions(df)
    st.pyplot(fig)
    
    # Nuvem de palavras
    st.subheader("☁️ Nuvem de Palavras (termos mais frequentes)")
    wc = generate_wordcloud(df)
    fig_wc, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig_wc)