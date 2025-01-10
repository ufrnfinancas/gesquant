#BDR
import pandas as pd
import yfinance as yf

# Monitor de Spread de BDR
'''
Há um spread, uma diferença entre o valor da ação e o de um lote de BDRs equivalente 
a uma ação (lembrando que um BDR não necessariamente corresponde a apenas um papel). 
Para calcular, você deve multiplicar o preço da ação pelo câmbio atual e pela proporção
de ações no lote. A diferença entre esse valor e o do BDR é o spread.
'''
bdr = '/home/vinicio/Dropbox/Fortem/GitHub/quant/indicebdr.xlsx'

dados = pd.read_excel(bdr)
dados['BDR'] = dados['BDR'] + '.SA'

usdbrl = yf.download('USDBRL=X', period='1d')['Close'].iloc[-1]

#Função de última cotação
def obter_ultima_cotacao(ticker):
    data = yf.download(ticker, period='1d')
    return data['Close'].iloc[-1]

dados['Ultima Cotacao BDR'] = dados['BDR'].apply(lambda x: obter_ultima_cotacao(x))
dados['Ultima Cotacao Acao'] = dados['Acao'].apply(lambda x: obter_ultima_cotacao(x))
dados2=dados
dados=dados2
# Calculando o spread
dados['Spread'] = dados['Prop'] * usdbrl * dados['Ultima Cotacao Acao'] - dados['Ultima Cotacao BDR']
dados

df_ordenado = dados.sort_values(by='Spread', ascending=False)

df = df_ordenado
# Reordenar colunas
df = df.reindex(columns=['Nome', 'BDR', 'Acao', 'Spread'])

df['BDR'] = df['BDR'].str.replace('.SA', '')
# Obter os 10 maiores spreads
maiores_spreads = df.head(10)

# Obter os 10 menores spreads
menores_spreads = df.tail(10)

maiores_spreads
menores_spreads

maiores_spreads.to_excel('/home/vinicio/Dropbox/Fortem/GitHub/quant/bdrmaiores.xlsx', index=False)
menores_spreads.to_excel('/home/vinicio/Dropbox/Fortem/GitHub/quant/bdrmenores.xlsx', index=False)

maiores_spreads.to_excel('/home/vinicio/Dropbox/Fortem/GitHub/harpa/bdrmaiores.xlsx', index=False)
menores_spreads.to_excel('/home/vinicio/Dropbox/Fortem/GitHub/harpa/bdrmenores.xlsx', index=False)






'''


import pandas as pd
import yfinance as yf
import matplotlib.pylab as plt
import requests
from urllib import request
from urllib.request import Request, urlopen


# A URL
url = "https://investnews.com.br/financas/veja-a-lista-completa-dos-bdrs-disponiveis-para-pessoas-fisicas-na-b3/"
request_site = Request(url, headers = {"User-Agent":"Mozilla/5.0"})
webpage = urlopen(request_site).read()

df = pd.read_html(webpage)[0]
df.columns = df.iloc[0]
df.drop([0], axis = 0, inplace = True)

bdrs = list(df['CÓDIGO'])
bdr_tickers = [i + '.SA' for i in bdrs]

dados = yf.download(bdr_tickers, start = '2023-06-01')['Close']

dados.dropna(axis=1, how='all')


def compara_ativos(inicio):
  dados = yf.download(['AAPL34.SA', 'AAPL'], start = inicio, progress = False)['Adj Close']
  dados = dados.dropna()
  dados = dados/dados.iloc[0]
  dados.plot()

compara_ativos('2018-06-01')
plt.show()

datas = ['2018-01-01', '2019-01-01','2020-01-01','2021-01-01','2022-01-01','2023-01-01']

dados = yf.download(['AAPL34.SA', 'AAPL', 'USDBRL=X'], start = '2017-01-03')['Adj Close']
dados = dados.ffill(axis = 0)
dados['Apple_reais'] = dados['AAPL'] * dados['USDBRL=X']

dados['Apple_reais'] = dados['AAPL'] * dados['USDBRL=X']
dados_normalizados.plot()

dados_normalizados.plot()

import plotly.express as px

fig = px.line(x=spread.index, y=spread)

fig.add_hline(y=spread.mean(), line_width=5, line_color="green")
fig.add_hline(y=(spread.mean()-spread.std()), line_width=3,
              line_dash="dash", line_color="orange")
fig.add_hline(y=(spread.mean()+spread.std()), line_width=3,
              line_dash="dash", line_color="orange")
fig.add_hline(y=(spread.mean()-2*spread.std()), line_width=5,
              line_dash="dash", line_color="red")
fig.add_hline(y=(spread.mean()+2*spread.std()), line_width=5,
              line_dash="dash", line_color="red")

fig.update_layout(xaxis_rangeslider_visible=False,
                  title_text='Razão entre preço AAPL em R$ e AAPL34 (entre 2018 e 2023)'

,template = 'simple_white',width=1100,height=500)

fig.show()

'''