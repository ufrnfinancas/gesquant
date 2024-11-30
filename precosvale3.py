import yfinance as yf
import matplotlib.pyplot as plt

# Define o ticker e o intervalo de datas
ticker = "VALE3.SA"
start_date = "2023-01-01"
end_date = "2024-11-27"  # Altere para a data atual, se necessário

# Baixa os dados históricos do yfinance
data = yf.download(ticker, start=start_date, end=end_date)

# Suponha a quantidade de ações adquiridas no início do período
initial_quantity = 50000  # Número de ações compradas inicialmente

# Calcula o valor inicial e final do investimento em ações
initial_price = data['Adj Close'].iloc[0]
final_price = data['Adj Close'].iloc[-1]
initial_investment_value = initial_quantity * initial_price
final_investment_value = initial_quantity * final_price

# Calcula o retorno no período
period_return = (final_price / initial_price) - 1

tickersemsa = ticker[:-3]
# Plota o gráfico de preços de fechamento ajustado
plt.figure(figsize=(14, 7))
plt.plot(data['Adj Close'], label=f'Preço de Fechamento de {tickersemsa}')
plt.title(f'Preço de {tickersemsa} de {start_date} a {end_date}')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento')
plt.grid(True)
plt.legend()
plt.show()


print(f"Retorno no período: {period_return:.2%}")
print(f"Valor inicial do investimento ({initial_quantity} ações): {initial_investment_value:.2f} BRL")
print(f"Valor final do investimento ({initial_quantity} ações): {final_investment_value:.2f} BRL")
