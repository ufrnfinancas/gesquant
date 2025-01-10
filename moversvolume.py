# Monitor de 5 dias

'''
Do Ibovespa:
- 10 ações que mais subiram e mais caíram nos últimos 5 dias úteis
- 10 ações com maior alta e maior baixa de volume como média nos últimos 5 dias úteis
- 10 ações com maior alta e maior baixa de volatilidade como média dos últimos 5 dias úteis 
'''

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Datas
end = datetime.now()
start = end - timedelta(days=60)

# fetch
ativos = ["ABEV3.SA", "ALPA4.SA", "ARZZ3.SA", "ASAI3.SA", "AZUL4.SA", "B3SA3.SA", "BBAS3.SA", 
          "BBDC3.SA", "BBDC4.SA", "BBSE3.SA", "BEEF3.SA", "BPAC11.SA", "BRAP4.SA", "BRFS3.SA", "BRKM5.SA", 
          "CASH3.SA", "CCRO3.SA", "CIEL3.SA", "CMIG4.SA", "CMIN3.SA", "COGN3.SA", "CPFE3.SA", "CPLE6.SA", 
          "CRFB3.SA", "CSAN3.SA", "CSNA3.SA", "CVCB3.SA", "CYRE3.SA", "DXCO3.SA", "EGIE3.SA", "ELET3.SA", 
          "ELET6.SA", "EMBR3.SA", "ENEV3.SA", "ENGI11.SA", "EQTL3.SA", "EZTC3.SA", "FLRY3.SA", 
          "GGBR4.SA", "GOAU4.SA", "GOLL4.SA", "HAPV3.SA", "HYPE3.SA", "IGTI11.SA", "IRBR3.SA", "ITSA4.SA", 
          "ITUB4.SA", "JBSS3.SA", "KLBN11.SA", "LREN3.SA", "LWSA3.SA", "MGLU3.SA", "MRFG3.SA", "MRVE3.SA", 
          "MULT3.SA", "NTCO3.SA", "PCAR3.SA", "PETR3.SA", "PETR4.SA", "PETZ3.SA", "PRIO3.SA", "RADL3.SA", 
          "RAIL3.SA", "RAIZ4.SA", "RDOR3.SA", "RENT3.SA", "RRRP3.SA", "SANB11.SA", "SBSP3.SA", "SLCE3.SA", 
          "SMTO3.SA", "SOMA3.SA", "SUZB3.SA", "TAEE11.SA", "TIMS3.SA", "TOTS3.SA", "UGPA3.SA", "USIM5.SA", 
          "VALE3.SA", "VBBR3.SA", "BHIA3.SA", "VIVT3.SA", "WEGE3.SA", "YDUQ3.SA"]


## Retornos

cotas = yf.download(ativos, start=start, end=end)["Adj Close"]

# Calcular os retornos percentuais diários
retornos_diarios = cotas.pct_change()

# Calcular a variação percentual nos últimos 5 dias úteis
var_percentual_5d = retornos_diarios.tail(5).mean()

# Identificar os 10 maiores e menores retornos
maiores_retornos = var_percentual_5d.nlargest(10)
menores_retornos = var_percentual_5d.nsmallest(10)

df_maiores_retornos = pd.DataFrame({'Ticker': maiores_retornos.index.str.replace('.SA', ''), 
                                    'Retorno (%)': (maiores_retornos.values * 100).round(2)})
df_menores_retornos = pd.DataFrame({'Ticker': menores_retornos.index.str.replace('.SA', ''), 
                                    'Retorno (%)': (menores_retornos.values * 100).round(2)})

df_maiores_retornos
df_menores_retornos

## Volume
volume = yf.download(ativos, start=start, end=end)["Volume"]

# Calcular os retornos percentuais diários
mudanca_diaria_volume = volume.pct_change()

# Calcular a variação percentual média nos últimos 5 dias úteis
var_percentual_5d_vol = mudanca_diaria_volume.tail(5).mean()

# Identificar os 10 maiores e menores retornos
volume_cresce = var_percentual_5d_vol.nlargest(10)
volume_cai = var_percentual_5d_vol.nsmallest(10)

df_maiores_altasvol = pd.DataFrame({'Ticker': volume_cresce.index.str.replace('.SA', ''), 
                                    'Mudança (%)': (volume_cresce.values * 100).round(2)})
df_maiores_quedasvol = pd.DataFrame({'Ticker': volume_cai.index.str.replace('.SA', ''), 
                                    'Mudança (%)': (volume_cai.values * 100).round(2)})

df_maiores_altasvol
df_maiores_quedasvol

## Agora mudança de volatilidade

# Calcular a volatilidade diária em uma janela móvel dos últimos 22 dias úteis para cada ação
volatilidade = cotas.pct_change().rolling(window=22).std()

diff_volatilidade = volatilidade.diff(periods=5)
maiores_crescimentos = diff_volatilidade.max().nlargest(10)
maiores_quedas = diff_volatilidade.min().nsmallest(10)
volatilidade.index = volatilidade.index.str.replace('.SA', '')
maiores_crescimentos.index = maiores_crescimentos.index.str.replace('.SA', '')
maiores_quedas.index = maiores_quedas.index.str.replace('.SA', '')

nomes_maiores_crescimentos = maiores_crescimentos.index.tolist()
nomes_maiores_quedas = maiores_quedas.index.tolist()

# Criar um novo DataFrame combinando os nomes das ações com os títulos das colunas
df_maiores_crescimentos_quedas = pd.DataFrame({
    "Maiores altas": nomes_maiores_crescimentos,
    "Maiores quedas": nomes_maiores_quedas
})

df_maiores_crescimentos_quedas