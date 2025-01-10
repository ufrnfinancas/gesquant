import yfinance as yf
import warnings
warnings.filterwarnings("ignore")
import riskfolio as rp
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta, time, datetime
import numpy as np

# Lista das ações
assets = [
    "ABEV3.SA", "ALPA4.SA", "ARZZ3.SA", "ASAI3.SA", "AZUL4.SA", "B3SA3.SA", "BBAS3.SA", 
    "BBDC3.SA", "BBDC4.SA", "BBSE3.SA", "BEEF3.SA", "BPAC11.SA", "BRAP4.SA", "BRFS3.SA", "BRKM5.SA", 
    "CASH3.SA", "CCRO3.SA", "CIEL3.SA", "CMIG4.SA", "CMIN3.SA", "COGN3.SA", "CPFE3.SA", "CPLE6.SA", 
    "CRFB3.SA", "CSAN3.SA", "CSNA3.SA", "CVCB3.SA", "CYRE3.SA", "DXCO3.SA", "EGIE3.SA", "ELET3.SA", 
    "ELET6.SA", "EMBR3.SA", "ENEV3.SA", "ENGI11.SA", "EQTL3.SA", "EZTC3.SA", "FLRY3.SA", 
    "GGBR4.SA", "GOAU4.SA", "GOLL4.SA", "HAPV3.SA", "HYPE3.SA", "IGTI11.SA", "IRBR3.SA", "ITSA4.SA", 
    "ITUB4.SA", "JBSS3.SA", "KLBN11.SA", "LREN3.SA", "LWSA3.SA", "MGLU3.SA", "MRFG3.SA", "MRVE3.SA", 
    "MULT3.SA", "NTCO3.SA", "PCAR3.SA", "PETR4.SA", "PETZ3.SA", "PRIO3.SA", "RADL3.SA", 
    "RAIL3.SA", "RAIZ4.SA", "RDOR3.SA", "RENT3.SA", "RRRP3.SA", "SANB11.SA", "SBSP3.SA", "SLCE3.SA", 
    "SMTO3.SA", "SOMA3.SA", "SUZB3.SA", "TAEE11.SA", "TIMS3.SA", "TOTS3.SA", "UGPA3.SA", "USIM5.SA", 
    "VALE3.SA", "VBBR3.SA", "BHIA3.SA", "VIVT3.SA", "WEGE3.SA", "YDUQ3.SA"
]

#download data
end = datetime.now()
start = end - timedelta(days = 180)
data = yf.download(assets, start=start, end=end)
# compute non-compounding, daily returns
returns = data['Adj Close'].pct_change().dropna()

# Portfolio with equal risk contribution weights
port = rp.Portfolio(returns=returns)
port.assets_stats(method_mu='hist', method_cov='hist', d=0.94)
w_rp = port.rp_optimization(
    model="Classic",  # use historical
    rm="MV",  # use mean-variance optimization
    hist=True,  # use historical scenarios
    rf=0,  # set risk free rate to 0
    b=None  # don't use constraints
)

# Portfolio with minimum return constraint
port.lowerret = 0.0019
# estimate the optimal portfolio with risk parity with the constraint
w_rp_c = port.rp_optimization(
    model="Classic",  # use historical
    rm="MV",  # use mean-variance optimization
    hist=True,  # use historical scenarios
    rf=0,  # set risk free rate to 0
    b=None  # don't use constraints
)

w_rp_c
