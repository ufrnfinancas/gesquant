import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuração da página
st.set_page_config(
    page_title="Cotações Ibovespa",
    page_icon="📈",
    layout="wide"
)

# Título principal
st.title("📈 Cotações do Ibovespa")

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações")

# Período de análise
periodo_opcoes = {
    "1 mês": "1mo",
    "3 meses": "3mo", 
    "6 meses": "6mo",
    "1 ano": "1y",
    "2 anos": "2y",
    "5 anos": "5y"
}

periodo_selecionado = st.sidebar.selectbox(
    "Selecione o período:",
    list(periodo_opcoes.keys()),
    index=3  # 1 ano como padrão
)

# Função para carregar dados
@st.cache_data
def carregar_dados_ibovespa(periodo):
    """Carrega dados do Ibovespa usando yfinance"""
    try:
        # ^BVSP é o símbolo do Ibovespa no Yahoo Finance
        ibov = yf.Ticker("^BVSP")
        dados = ibov.history(period=periodo)
        return dados
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

# Carregar dados
with st.spinner("Carregando dados do Ibovespa..."):
    dados = carregar_dados_ibovespa(periodo_opcoes[periodo_selecionado])

if dados is not None and not dados.empty:
    
    # Informações básicas
    col1, col2, col3, col4 = st.columns(4)
    
    ultimo_preco = dados['Close'].iloc[-1]
    variacao = dados['Close'].iloc[-1] - dados['Close'].iloc[-2]
    variacao_pct = (variacao / dados['Close'].iloc[-2]) * 100
    
    with col1:
        st.metric(
            label="Último Preço",
            value=f"{ultimo_preco:,.0f}",
            delta=f"{variacao:+,.0f}"
        )
    
    with col2:
        st.metric(
            label="Variação %",
            value=f"{variacao_pct:+.2f}%"
        )
    
    with col3:
        st.metric(
            label="Máxima (período)",
            value=f"{dados['High'].max():,.0f}"
        )
    
    with col4:
        st.metric(
            label="Mínima (período)",
            value=f"{dados['Low'].min():,.0f}"
        )
    
    # Gráfico de linha
    st.subheader("📈 Gráfico de Preços")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dados.index,
        y=dados['Close'],
        mode='lines',
        name='Ibovespa',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="Evolução do Ibovespa",
        xaxis_title="Data",
        yaxis_title="Pontos",
        height=600,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Dados recentes
    st.subheader("📅 Últimos 10 Pregões")
    dados_recentes = dados.tail(10)[['Open', 'High', 'Low', 'Close']].copy()
    dados_recentes.columns = ['Abertura', 'Máxima', 'Mínima', 'Fechamento']
    
    # Formatar números
    for col in dados_recentes.columns:
        dados_recentes[col] = dados_recentes[col].round(0).astype(int)
    
    st.dataframe(dados_recentes, use_container_width=True)

else:
    st.error("Não foi possível carregar os dados do Ibovespa. Verifique sua conexão com a internet.")
    
# Rodapé
st.markdown("---")
st.markdown("💡 **Fonte de dados:** Yahoo Finance via yfinance")