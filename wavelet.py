import numpy as np
import pywt
import matplotlib.pyplot as plt
import yfinance as yf
import matplotlib.dates as mdates

def wavelet_decomposition(data, wavelet='haar', level=4):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    return coeffs

def plot_wavelet_decomposition(data, coeffs, dates, wavelet='haar'):
    fig, axarr = plt.subplots(len(coeffs) + 1, sharex=True, figsize=(10, 6))
    axarr[0].plot(dates, data, color='blue')
    axarr[0].set_title('Original Signal')
    axarr[0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))  # Display major ticks every January and July
    axarr[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format x-axis to display only year-month
    axarr[0].tick_params(axis='x', rotation=90)  # Rotate x-axis labels vertically

    for i, coeff in enumerate(coeffs):
        axarr[i+1].plot(dates[:len(coeff)], coeff, color='red')
        axarr[i+1].set_title(f'Decomposition Coefficients (Level {i+1})')
        axarr[i+1].tick_params(axis='x', rotation=90)  # Rotate x-axis labels vertically

    # Automatic crash flagging
    threshold = 0.05  # Define a threshold for crash detection (you can adjust this)
    for i, coeff in enumerate(coeffs[:2]):  # Check detail coefficients from levels 1 and 2
        if max(abs(coeff)) > threshold:  # If maximum coefficient exceeds the threshold
            axarr[i+2].axhline(y=max(abs(coeff)), color='orange', linestyle='--')  # Plot a dashed line to indicate potential crash

    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

# Download data from Yahoo Finance
start = '2005-05-01'
end = '2024-03-19'
ibov = yf.download('^BVSP', period='1d', start=start, end=end)
retorno = ibov['Adj Close'].pct_change().dropna()  # Calculate daily returns, drop NaN values
data = retorno.values  # Convert to numpy array
dates = retorno.index  # Get dates

# Perform wavelet decomposition
coeffs = wavelet_decomposition(data)

# Plot original signal and decomposition coefficients
plot_wavelet_decomposition(data, coeffs, dates)


'''
Nível 1 (Detalhe 1):
Este nível representa os detalhes de alta frequência da série temporal, capturando 
variações rápidas e pequenas mudanças nos dados. Em uma série financeira, essas variações 
podem corresponder a flutuações de curto prazo ou ruído de mercado.

Nível 2 (Detalhe 2):
Similar ao Nível 1, mas com uma frequência ainda mais alta. Este nível pode capturar 
flutuações ainda menores e é mais sensível a mudanças rápidas nos dados.

Nível 3 (Detalhe 3):
Neste nível, você começa a capturar padrões de frequência intermediária na série temporal. 
Isso pode incluir variações que ocorrem em uma escala de tempo um pouco mais longa do que 
nos níveis anteriores.

Nível 4 (Detalhe 4):
Este nível representa padrões de frequência mais baixa e variações de médio prazo na série 
temporal. Pode capturar mudanças de tendência e ciclos de mercado que ocorrem em escalas de 
tempo mais amplas.

Nível 5 (Aproximação 5):
No último nível de decomposição, você obtém uma aproximação suavizada da série temporal 
original. Este componente representa as tendências de longo prazo e as variações lentas 
na série.


Ao analisar os resultados da decomposição wavelet nos cinco níveis, você pode observar 
como os diferentes componentes da série temporal contribuem para sua variabilidade total 
em diferentes escalas de tempo. Isso pode ajudar na identificação de padrões, ciclos e 
anomalias nos dados financeiros, incluindo potenciais quedas no mercado de capitais. 
Por exemplo, uma mudança abrupta nos coeficientes de detalhe de alta frequência nos níveis 
1 e 2 pode indicar volatilidade aumentada ou eventos de curto prazo no mercado, enquanto 
mudanças nos níveis de aproximação e detalhe de baixa frequência podem sugerir mudanças 
nas tendências de longo prazo.
'''