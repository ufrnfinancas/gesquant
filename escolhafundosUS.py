import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

##### HSA

### Análise preliminar
fundoshsa = [
    'WSCRX', 'WGIFX', 'RIGGX', 'BRHYX', 'CSMUX', 'CSFZX', 'COFYX', 'CSVYX',
    'DCZRX', 'SEMTX', 'BDOKX', 'BDBKX', 'BRMKX', 'WFSPX', 'WFBIX', 'MLAIX',
    'MEIKX', 'NSRIX', 'PCRIX', 'PTLDX', 'PTTRX', 'VTWAX', 'VMFXX', 'VFTAX',
    'VAIPX', 'VASGX', 'VSCGX', 'VSMGX', 'VTWNX', 'VTTVX', 'VTHRX', 'VTTHX',
    'VFORX', 'VTIVX', 'VFIFX', 'VFFVX', 'VTTSX', 'VLXVX', 'VSVNX', 'VTINX'
]

# Verificar fundos disponíveis
fundos_validos = []
for ticker in fundoshsa:
    df = yf.download(ticker, period="5y", interval="1mo", auto_adjust=True)['Close']
    if not df.empty:
        fundos_validos.append(ticker)

print(f"Fundos com dados disponíveis: {fundos_validos}")


# Baixar preços ajustados de fechamento dos últimos 5 anos
dados_mensal = yf.download(fundos_validos, period="5y", interval="1mo", auto_adjust=True)['Close']


# Calcular retorno anualizado
retorno_anual = ((dados_mensal.iloc[-1] / dados_mensal.iloc[0]) ** (1/5)) - 1
retorno_anual.dropna().sort_values(ascending=False).apply(lambda x: f"{x:.2%}")

# Exibir resultados ordenados
retorno_anual.sort_values(ascending=False).apply(lambda x: f"{x:.2%}")

# Plot do Top 10
top10 = retorno_anual.dropna().sort_values(ascending=False).head(10)
top10.plot(kind='bar', figsize=(10, 5), color='green')
plt.title('Top 10 Fundos – Retorno Anualizado (5 anos)')
plt.ylabel('Retorno Anual')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

### Agora a análise com dados mensais
dados = yf.download(fundos_validos, period="5y", interval="1mo", auto_adjust=True)['Close']
dados = dados.dropna(axis=1, how='any')  # Remove fundos com dados faltando
ret_mensal = dados.pct_change().dropna()
vol_anual = ret_mensal.std() * np.sqrt(12)

def max_drawdown_mensal(serie_ret):
    acm = (1 + serie_ret).cumprod()
    max_acm = acm.cummax()
    drawdown = (acm - max_acm) / max_acm
    return drawdown.min()

max_dd = ret_mensal.apply(max_drawdown_mensal)

ret_anual = (1 + ret_mensal.mean())**12 - 1
sharpe = ret_anual / vol_anual


# Finalmente, os plots
fig, axs = plt.subplots(3, 1, figsize=(14, 12))

# Volatilidade
vol_anual.sort_values(ascending=False).head(10).plot(kind='bar', ax=axs[0], color='darkorange')
axs[0].set_title('Top 10 – Volatilidade Anualizada (Mensal)')
axs[0].set_ylabel('Volatilidade')

# Drawdown
max_dd.sort_values().head(10).plot(kind='bar', ax=axs[1], color='crimson')
axs[1].set_title('Top 10 – Menor Máximo Drawdown (Mensal)')
axs[1].set_ylabel('Drawdown')

# Sharpe
sharpe.sort_values(ascending=False).head(10).plot(kind='bar', ax=axs[2], color='seagreen')
axs[2].set_title('Top 10 – Sharpe Ratio Anualizado (Mensal)')
axs[2].set_ylabel('Sharpe')

plt.tight_layout()
plt.show()

# A tabela

# Retorno anualizado
ret_anual = (1 + ret_mensal.mean())**12 - 1

# Consolidar
df_metrics = pd.DataFrame({
    'Retorno Anual (%)': ret_anual * 100,
    'Volatilidade Anual (%)': vol_anual * 100,
    'Sharpe Ratio': sharpe,
    'Max Drawdown (%)': max_dd * 100
})

# Arredondar
df_metrics = df_metrics.round(2)

retorno_ordenado = df_metrics.sort_values(by='Retorno Anual (%)', ascending=False)
sharpe_ordenado = df_metrics.sort_values(by='Sharpe Ratio', ascending=False)
drawdown_ordenado = df_metrics.sort_values(by='Max Drawdown (%)', ascending=False)

# Matriz de correlação
correlacao = ret_mensal.corr()

#
##
###
####
##### 401k

# Verificar fundos disponíveis
fundos401 = [
    "PTUIX", "VIPIX", "VBTIX", "WATFX", "AACWIX", "MADVX", "MAFOX", "MASKX", "DOXGX", "HMCNX", 
    "MGRAX", "NSRIX", "SSRSX", "TRLGX", "VFFSX", "VIEIX", "VTSNX", "BACIX", 
    "MALOX", "LINIX", "LIJIX", "LIKIX", "LIHIX", "LIPIX", "LIVIX", "LIZKX", "LIWIX", "LIYIX", "LIRIX", "PAAIX",
    "VMFXX"
]

fundos_validos = []
fundos_invalidos = []

for ticker in fundos401:
    df = yf.download(ticker, period="5y", interval="1mo", auto_adjust=True)['Close']
    if not df.empty:
        fundos_validos.append(ticker)
    else:
        fundos_invalidos.append(ticker)

print(f"Fundos com dados disponíveis: {fundos_validos}")
print(f"Fundos sem dados disponíveis: {fundos_invalidos}")

# Baixar preços ajustados de fechamento dos últimos 5 anos
dados_mensal = yf.download(fundos_validos, period="5y", interval="1mo", auto_adjust=True)['Close']

# Calcular retorno anualizado
retorno_anual = ((dados_mensal.iloc[-1] / dados_mensal.iloc[0]) ** (1/5)) - 1
retorno_anual.dropna().sort_values(ascending=False).apply(lambda x: f"{x:.2%}")

# Exibir resultados ordenados
retorno_anual.sort_values(ascending=False).apply(lambda x: f"{x:.2%}")

# Plot do Top 10
top10 = retorno_anual.dropna().sort_values(ascending=False).head(10)
top10.plot(kind='bar', figsize=(10, 5), color='green')
plt.title('Top 10 Fundos – Retorno Anualizado (5 anos)')
plt.ylabel('Retorno Anual')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

### Agora a análise com dados mensais
dados = yf.download(fundos_validos, period="5y", interval="1mo", auto_adjust=True)['Close']
dados = dados.dropna(axis=1, how='any')  # Remove fundos com dados faltando
ret_mensal = dados.pct_change().dropna()
vol_anual = ret_mensal.std() * np.sqrt(12)

def max_drawdown_mensal(serie_ret):
    acm = (1 + serie_ret).cumprod()
    max_acm = acm.cummax()
    drawdown = (acm - max_acm) / max_acm
    return drawdown.min()

max_dd = ret_mensal.apply(max_drawdown_mensal)

ret_anual = (1 + ret_mensal.mean())**12 - 1
sharpe = ret_anual / vol_anual


# Finalmente, os plots
fig, axs = plt.subplots(3, 1, figsize=(14, 12))

# Volatilidade
vol_anual.sort_values(ascending=False).head(10).plot(kind='bar', ax=axs[0], color='darkorange')
axs[0].set_title('Top 10 – Volatilidade Anualizada (Mensal)')
axs[0].set_ylabel('Volatilidade')

# Drawdown
max_dd.sort_values().head(10).plot(kind='bar', ax=axs[1], color='crimson')
axs[1].set_title('Top 10 – Menor Máximo Drawdown (Mensal)')
axs[1].set_ylabel('Drawdown')

# Sharpe
sharpe.sort_values(ascending=False).head(10).plot(kind='bar', ax=axs[2], color='seagreen')
axs[2].set_title('Top 10 – Sharpe Ratio Anualizado (Mensal)')
axs[2].set_ylabel('Sharpe')

plt.tight_layout()
plt.show()

# A tabela

# Retorno anualizado
ret_anual = (1 + ret_mensal.mean())**12 - 1

# Consolidar
df_metrics = pd.DataFrame({
    'Retorno Anual (%)': ret_anual * 100,
    'Volatilidade Anual (%)': vol_anual * 100,
    'Sharpe Ratio': sharpe,
    'Max Drawdown (%)': max_dd * 100
})

# Arredondar
df_metrics = df_metrics.round(2)

retorno_ordenado = df_metrics.sort_values(by='Retorno Anual (%)', ascending=False)
sharpe_ordenado = df_metrics.sort_values(by='Sharpe Ratio', ascending=False)
drawdown_ordenado = df_metrics.sort_values(by='Max Drawdown (%)', ascending=False)

# Matriz de correlação
correlacao = ret_mensal.corr()

