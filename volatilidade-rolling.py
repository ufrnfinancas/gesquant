# volatilidade-rolling
import datetime
from datetime import datetime, timedelta, time
import yfinance as yf
import pandas as pd

end = datetime.now()
start = end - timedelta(days=90)

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

# Calcular a volatilidade diária em uma janela móvel dos últimos 22 dias úteis para cada ação
cotas = yf.download(ativos, start=start, end=end)["Adj Close"]

# Calcular os retornos diários das ações
retornos_diarios = cotas.pct_change()

dp1 = retornos_diarios.tail(22).std()
dp1 = pd.DataFrame(dp1)
dp1 = dp1.T
retornos_diarios.drop(retornos_diarios.tail(1).index,inplace = True)

for i in range(4):
    dpx = retornos_diarios.tail(22).std()
    dpx = pd.DataFrame(dpx)
    dpx = dpx.T
    dp1 = pd.concat([dp1, dpx])
    retornos_diarios.drop(retornos_diarios.tail(1).index,inplace = True)

diffvol = dp1.pct_change()
mediasvol = diffvol.mean()
maisvol = mediasvol.nlargest(10)
menosvol = mediasvol.nsmallest(10)

maisvol = pd.DataFrame({'Ticker': maisvol.index.str.replace('\.SA', ''), 
                                    'Diferença (%)': (maisvol.values * 100)})
menosvol = pd.DataFrame({'Ticker': menosvol.index.str.replace('\.SA', ''), 
                                    'Diferença (%)': (menosvol.values * 100)})

maisvol['Diferença (%)'] = maisvol['Diferença (%)'].apply(lambda x: '{:.2f}'.format(x))
menosvol['Diferença (%)'] = menosvol['Diferença (%)'].apply(lambda x: '{:.2f}'.format(x))

























