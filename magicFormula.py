import fundamentus
import pandas as pd

dfraw = fundamentus.get_resultado_raw()
print(dfraw.columns)
df = fundamentus.get_resultado()
print(df.columns)
# 'Cotação', 'P/L', 'P/VP', 'PSR', 'Div.Yield', 'P/Ativo', 'P/Cap.Giro',
#       'P/EBIT', 'P/Ativ Circ.Liq', 'EV/EBIT', 'EV/EBITDA', 'Mrg Ebit',
#       'Mrg. Líq.', 'Liq. Corr.', 'ROIC', 'ROE', 'Liq.2meses', 'Patrim. Líq',
#       'Dív.Brut/ Patrim.', 'Cresc. Rec.5a'],
# 'cotacao', 'pl', 'pvp', 'psr', 'dy', 'pa', 'pcg', 'pebit', 'pacl',
#       'evebit', 'evebitda', 'mrgebit', 'mrgliq', 'roic', 'roe', 'liqc',
#       'liq2m', 'patrliq', 'divbpatr', 'c5y'
# Primeiros filtros
df2 = df[df.pl > 0]
df3 = df2[df2.evebit > 0]
df4 = df3[df3.roic > 0]
df5 = df4[df4.patrliq > 100000000]
stocks_df = df5[df5.liq2m > 0]
stocks_df = stocks_df[['evebit', 'roic']]
# Trabalhando apenas com dy e roic
data = {'Stock': stocks_df.index,
        'Earnings_Yield': stocks_df['evebit'],
        'ROIC': stocks_df['roic']}
stocks_df = pd.DataFrame(data)
# Ordenando dy e roic
stocks_df['Earnings_Yield_Rank'] = stocks_df['Earnings_Yield'].rank(ascending=True)
stocks_df['ROIC_Rank'] = stocks_df['ROIC'].rank(ascending=False)
# Calculando a magic formula
stocks_df['Magic_Formula_Rank'] = stocks_df['Earnings_Yield_Rank'] + stocks_df['ROIC_Rank']
# Ordenando pela magic formula
sorted_stocks = stocks_df.sort_values('Magic_Formula_Rank')
# Visualizando
print(sorted_stocks[['Stock', 'Earnings_Yield', 'ROIC', 'Magic_Formula_Rank']])
ativos = sorted_stocks.head(30)
ativos

# Removendo baixa liquidez
codigos_a_remover = ['CEDO4', 'RSUL4', 'CEDO3', 'CAMB3', 'PETR3', 'MTSA4', 'DEXP3', 'MRSA6B', 'AURA33']
ativos = ativos.drop(codigos_a_remover, axis=0)
ativos.head(10)
carteiramf = ativos.head(10)