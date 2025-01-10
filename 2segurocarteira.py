# Seguro da carteira

import pandas as pd
import requests
import yfinance as yf
from openpyxl import Workbook, load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
# Option chain de BOVA11
subjacente = 'BOVA11'   # não vem do Yfinance! esqueça o ".SA" ao final do ticker!
vencimento = '2024-10-18'      # YYYY-MM-DD
def optionchaindate(subjacente, vencimento):
    url = f'https://opcoes.net.br/listaopcoes/completa?idAcao={subjacente}&listarVencimentos=false&cotacoes=true&vencimentos={vencimento}'
    r = requests.get(url).json()
    x = ([subjacente, vencimento, i[0].split('_')[0], i[2], i[3], i[5], i[8], i[9], i[10]] for i in r['data']['cotacoesOpcoes'])
    return pd.DataFrame(x, columns=['subjacente', 'vencimento', 'ativo', 'tipo', 'modelo', 'strike', 'preco', 'negocios', 'volume'])

chain = optionchaindate(subjacente, vencimento)

# Filtros
symbol = 'BOVA11.SA'
stock_data = yf.download(symbol)['Adj Close']
last_price = stock_data.iloc[-1]
chain2 = chain[chain['tipo'] == 'PUT']
chain3 = chain2[chain2['strike'] < last_price*0.85]
chain4 = chain3.sort_values(by='strike', ascending=False)
chain5 = chain4.drop('modelo', axis=1)
chain6 = chain5.reset_index(drop=True)
bova11 = chain6.head(10)

# Renomeie as colunas
novo_nome_colunas = {
    'subjacente': 'Subjacente',
    'vencimento': 'Vencimento',
    'ativo': 'Ativo',
    'tipo': 'Tipo',
    'strike': 'Strike',
    'preco': 'Preço',
    'negocios': 'Negócios',
    'volume': 'Volume'
}

bova11 = bova11.rename(columns=novo_nome_colunas)

wb = Workbook()
ws = wb.active

# Adicione os dados do DataFrame ao arquivo Excel
for row in dataframe_to_rows(bova11, index=False, header=True):
    ws.append(row)

# Salve o arquivo Excel
wb.save('C:/repo/harpa/bova11_disaster.xlsx')

wb.save('C:/repo/quant/bova11_disaster.xlsx')