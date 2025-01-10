#BDR

import pandas as pd
import yfinance as yf

bdr = 'C:/repo/quant/indicebdr.xlsx' 

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

maiores_spreads.to_excel('C:/repo/quant/bdrmaiores.xlsx', index=False)
menores_spreads.to_excel('C:/repo/quant/bdrmenores.xlsx', index=False)

maiores_spreads.to_excel('C:/repo/harpapro/bdrmaiores.xlsx', index=False)
menores_spreads.to_excel('C:/repo/harpapro/bdrmenores.xlsx', index=False)
