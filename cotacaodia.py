import yfinance as yf

# Defina o símbolo da ação e a data desejada
ticker = "VALE3.SA"
data_especifica = "2024-06-17"  # Exemplo de data
# Baixar os dados da ação
acao = yf.download(ticker, start=data_especifica)
acao.head(3)

# Defina o símbolo da ação e a data desejada
ticker = "VALE3.SA"
data_especifica = "2024-06-21"  # Exemplo de data
# Baixar os dados da ação
acao = yf.download(ticker, start=data_especifica)
acao.head(3)

