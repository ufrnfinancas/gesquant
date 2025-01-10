# Curva de juros
import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import matplotlib.pyplot as plt


# Inicializa o driver do navegador (sem especificar o caminho do executável)
driver = webdriver.Chrome()

# Abre o site
driver.get('https://www2.bmf.com.br/pages/portal/bmfbovespa/lumis/lum-boletim-online-ptBR.asp?Acao=BUSCA&cboMercadoria=DI1')
driver.find_element(By.XPATH, '//*[@id="principal"]')
codigo_tabela = driver.find_element(By.XPATH, '//*[@id="principal"]/tbody').get_attribute('outerHTML')

tabela1 = pd.read_html(codigo_tabela, decimal = ',', thousands = '.')[0]
tabela_full = tabela1

for i in range(0,3):
    tabela_full = pd.concat([tabela_full, pd.read_html(codigo_tabela, decimal = ',', thousands = '.')[i]], axis=1)

percent_missing = tabela_full.isnull().sum() * 100 / len(tabela_full)
tabela_full.drop(list(percent_missing[percent_missing==100].index), axis = 1, inplace = True)

tabela_full

# Plot
tabela_full.index = tabela_full['Vecto.']
tabela_full.index = tabela_full.iloc[:,1] #por conta da duplicacao da coluna de vencimento

tabela_full.iloc[:-1]['Último Preço'].plot(figsize = (12,6))
plt.show()
