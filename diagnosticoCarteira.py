# Diagnostico de carteira #

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

##### Carteira x Benchmark

assets = ['ABEV3.SA', 'CIEL3.SA', 'COGN3.SA', 'EGIE3.SA', 'KLBN11.SA', 
          'LWSA3.SA', 'MGLU3.SA', 'MRFG3.SA', 'MULT3.SA', 'PETZ3.SA']

start = '2024-01-02'

# Coleta de dados
prices = yf.download(assets, start = start)['Adj Close']

# Alocação na carteira em $
allocation = {'ABEV3.SA': 48507, 'CIEL3.SA': 45630, 'COGN3.SA': 45220, 'EGIE3.SA': 49716, 'KLBN11.SA': 52382, 
              'LWSA3.SA': 46008, 'MGLU3.SA': 30528, 'MRFG3.SA': 47771, 'MULT3.SA': 49140, 'PETZ3.SA': 37760}

sum(allocation.values())
first = prices.iloc[0]
allocationdf = pd.Series(data = allocation, index = list(allocation.keys()))

# Alocação na carteira em quantidade
qalloc = allocationdf / first
aum = prices * qalloc 

# Evolução de AUM
aum['AUM Total'] = aum.sum(axis = 1)

# Agora com o benchmark
bench = yf.download('^BVSP', start = start)
bench.rename(columns = {'Adj Close': 'Ibov'}, inplace = True)
bench.drop(bench.columns[[0,1,2,3,5]], axis = 1, inplace = True)
aumbench = pd.merge(bench, aum, how = 'inner', on = 'Date')
aumbenchnorm = aumbench / aumbench.iloc[0]

# Plot da carteira e do benchmark
aumbenchnorm[['Ibov', 'AUM Total']].plot(figsize = (10, 10))
plt.xlabel('Data')
plt.ylabel('Preço')
plt.title('AUM / Ibov / Normalizado')
plt.legend(fontsize = 5)
plt.show()


##### Desempenho

assets = ['ABEV3.SA', 'CIEL3.SA', 'COGN3.SA', 'EGIE3.SA', 'KLBN11.SA', 
          'LWSA3.SA', 'MGLU3.SA', 'MRFG3.SA', 'MULT3.SA', 'PETZ3.SA' ]
ibovdata = ['^BVSP']

# Coleta de dados
data = yf.download(assets, period = '1d', start = start)
ibov = yf.download(ibovdata, period = '1d', start = start)

# Normalizando os preços
datanorm = data['Adj Close']/data['Adj Close'].iloc[0]
# Remove the specified string from column titles
string_to_remove = '.SA'
datanorm.rename(columns=lambda x: x.replace(string_to_remove, ''), inplace=True)

# Plot
datanorm.plot()
font_type = "Arial"  # Replace with the font you want to use
font_size = 11
plt.ylabel('Preço')
plt.xlabel('Data')
plt.title('Performance de Ativos 2024.1 - Normalizado')
plt.legend(fontsize = 6)
ibovnorm = ibov['Adj Close']/ibov['Adj Close'].iloc[0]
ibovnorm.plot(label='IBOV', color='yellow')
plt.legend(fontsize = 6)
plt.xlabel('Data')
plt.show()


##### Drawdown

arquivo = 'cotas.xlsx'
dados = pd.read_excel(arquivo)
dados.set_index('Data', inplace=True)
returns = dados["Cota"].pct_change()

def drawdown(returns):
    """Determines the drawdown
    Parameters
    ----------
    returns : pd.Series
        Daily returns of an asset, noncumulative
    Returns
    -------
    drawdown : pd.Series
    """
    # replace the first nan value with 0.0
    returns.fillna(0.0, inplace=True)
    # create cumulative returns
    cumulative = (returns + 1).cumprod()
    # np.maximum.accumulate takes the running max value
    # of the input series. in this case, it will maintain
    # the running maximum value. this is the running
    # maximum return
    running_max = np.maximum.accumulate(cumulative)
    # compute the change between the cumulative return
    # and the running maximum return
    return (cumulative - running_max) / running_max

drawdown(returns).plot(kind="area", color="salmon", alpha=0.5)
plt.show()

def max_drawdown(returns):
    """ Determines the maximum drawdown
    Parameters
    ----------
    returns : pd.Series
        Daily returns of an asset, noncumulative
    Returns
    -------
    max_drawdown : float
    """
    return np.min(drawdown(returns))

returns.rolling(30).apply(max_drawdown).plot(kind="area", color="salmon", alpha=0.5)
plt.show()

##### V@R

assets = ['ABEV3.SA', 'CIEL3.SA', 'COGN3.SA', 'EGIE3.SA', 'KLBN11.SA', 
          'LWSA3.SA', 'MGLU3.SA', 'MRFG3.SA', "MULT3.SA", "PETZ3.SA" ]
weights = np.array([0.10, 0.10, 0.10, 0.10, 0.10,
                    0.10, 0.10, 0.10, 0.10, 0.10])

start = '2022-01-01'
portfolio = yf.download(assets, start = start)['Adj Close']
returns = portfolio.pct_change()
portfolioreturn = (returns * weights).sum(axis=1)

# Dataframe
portfolioreturndf = pd.DataFrame()
portfolioreturndf["Returns"] = portfolioreturn
portfolioreturndf.head()

# V@R
confidence_level = 0.95
historical_var = np.nanpercentile(portfolioreturndf, (1-confidence_level)*100)
print(f"VaR histórico a {confidence_level*100}% de confiança: {historical_var:.4f}")

#Plot
plt.hist(portfolioreturndf, bins=10, density=True, alpha=0.6, color='g', label='Retornos')
plt.axvline(x=historical_var, color='r', linestyle='--', label=f'{confidence_level*100}% VaR') # Lara delimitar o VaR
plt.xlabel('Retornos')
plt.ylabel('Frequência')
plt.legend()
plt.title('Cálculo do Value at Risk (VaR) Histórico')
plt.show()

##### PCA



# The data
symbols = ['PETR4.SA',
           'VALE3.SA',
           'BBAS3.SA',
           'CMIN3.SA',
           'EMBR3.SA']

data = yf.download(symbols, start = '2020-01-01', end = '2022-11-30')
portfolio_returns = data['Adj Close'].pct_change().dropna()

# The PCA model fit
pca = PCA(n_components=3)
pca.fit(portfolio_returns)

pct = pca.explained_variance_ratio_
pca_components = pca.components_

# The components plot
cum_pct = np.cumsum(pct)
x = np.arange(1,len(pct)+1,1)
plt.subplot(1, 2, 1)
plt.bar(x, pct * 100, align="center")
plt.title('Contribution (%)')
plt.xlabel('Component')
plt.xticks(x)
plt.xlim([0, 4])
plt.ylim([0, 100])
plt.subplot(1, 2, 2)
plt.plot(x, cum_pct * 100, 'ro-')
plt.title('Cumulative contribution (%)')
plt.xlabel('Component')
plt.xticks(x)
plt.xlim([0, 4])
plt.ylim([0, 100])
plt.show()

# Isolate the alpha factors
X = np.asarray(portfolio_returns)
factor_returns = X.dot(pca_components.T)
factor_returns = pd.DataFrame(
    columns=["f1", "f2", "f3"], 
    index=portfolio_returns.index,
    data=factor_returns
)
factor_returns.head()

factor_exposures = pd.DataFrame(
    index=["f1", "f2", "f3"], 
    columns=portfolio_returns.columns,
    data = pca_components
).T
factor_exposures.f1.sort_values().plot.bar()
plt.show()

# Another plot
labels = factor_exposures.index
data = factor_exposures.values
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('factor exposure of PC1')
plt.ylabel('factor exposure of PC2')
for label, x, y in zip(labels, data[:, 0], data[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), 
        xytext=(-20, 20),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
plt.show()

