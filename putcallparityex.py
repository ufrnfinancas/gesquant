### Put-Call Parity
import MetaTrader5 as mt5
import numpy as np

mt5.initialize()
 
print(mt5.terminal_info())
print(mt5.version())

####### Função de paridade put call
def putcall(ativo, call, put, k):
    r = 0.099355  # taxa de juros anual na curva 
    t = 23 / 252  # vencimento em dias/252
    cost = (
        4.90 * 2 #corretagem
    )  # custos de transacao, corretagens, emolumentos, aluguel
    lot = 100  # tamanho lote por operacao
    deviation = 2  # cada ponto, um centavo

    mt5.initialize()
    mt5.symbol_select(ativo, True)
    mt5.symbol_select(call, True)
    mt5.symbol_select(put, True)

    scan = 0
    abre = 0
    tentativain = 0

    while scan < 1:
        # time.sleep(1)
        lasttickativo = mt5.symbol_info_tick(ativo)
        lasttickcall = mt5.symbol_info_tick(call)
        lasttickput = mt5.symbol_info_tick(put)
        precob = lasttickativo.bid # vender no preco bid
        precoa = lasttickativo.ask # comprar no preco ask
        callb = lasttickcall.bid
        calla = lasttickcall.ask
        putb = lasttickput.bid
        puta = lasttickput.ask
        uLEE = (callb + k * np.exp(-r * t)) - (
            puta + precoa
        )  # para buscar diferenca positiva do (lado esquerdo - lado direito)
        uLDE = (putb + precob) - (
            calla + k * np.exp(-r * t)
        )  # para buscar diferenca positiva do (lado direito - lado esquerdo)

        if uLEE > 0 + cost:
            print("vende call, compra ativo e put")
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": call,
                "volume": float(lot),
                "type": mt5.ORDER_TYPE_SELL,
                "price": callb,
                "deviation": deviation,
                "magic": 123456,
                "comment": "venda",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            result = mt5.order_send(request)
            callbe = result[4]
            print(result)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": put,
                "volume": float(lot),
                "type": mt5.ORDER_TYPE_BUY,
                "price": puta,
                "deviation": deviation,
                "magic": 123456,
                "comment": "compra",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            result = mt5.order_send(request)
            putae = result[4]
            print(result)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": ativo,
                "volume": float(lot),
                "type": mt5.ORDER_TYPE_BUY,
                "price": precoa,
                "deviation": deviation,
                "magic": 123456,
                "comment": "compra",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
            result = mt5.order_send(request)
            precoae = result[4]
            print(result)
            abre = 1
            scan = 1
        else:
            if uLDE > 0 + cost:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": call,
                    "volume": float(lot),
                    "type": mt5.ORDER_TYPE_BUY,
                    "price": calla,
                    "deviation": deviation,
                    "magic": 123456,
                    "comment": "compra",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }
                result = mt5.order_send(request)
                callae = result[4]
                print(result)
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": put,
                    "volume": float(lot),
                    "type": mt5.ORDER_TYPE_SELL,
                    "price": putb,
                    "deviation": deviation,
                    "magic": 123456,
                    "comment": "venda",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }
                result = mt5.order_send(request)
                putbe = result[4]
                print(result)
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": ativo,
                    "volume": float(lot),
                    "type": mt5.ORDER_TYPE_SELL,
                    "price": precob,
                    "deviation": deviation,
                    "magic": 123456,
                    "comment": "venda",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }
                result = mt5.order_send(request)
                precobe = result[4]
                print(result)
                print("compra call, vende ativo e put")
                abre = -1
                scan = 1
            else:
                print("Sem oportunidade de abrir")
                abre = 0
                scan = 0
                tentativain = tentativain + 1
                print(tentativain)

    # Saida da carteira
    stop = 0
    tentativaout = 0
    while stop < 1:
        mt5.initialize()
        mt5.symbol_select(ativo, True)
        mt5.symbol_select(call, True)
        mt5.symbol_select(put, True)
        lasttickativo = mt5.symbol_info_tick(ativo)
        lasttickcall = mt5.symbol_info_tick(call)
        lasttickput = mt5.symbol_info_tick(put)
        precob = lasttickativo.bid
        precoa = lasttickativo.ask
        callb = lasttickcall.bid
        calla = lasttickcall.ask
        putb = lasttickput.bid
        puta = lasttickput.ask
        uLEE = (callb + k * np.exp(-r * t)) - (
            puta + precoa
        )  # para buscar diferenca positiva do lado esquerdo - lado direito
        uLDE = (putb + precob) - (
            calla + k * np.exp(-r * t)
        )  # para buscar diferenca positiva do lado direito - lado esquerdo
        if abre > 0:  # aberto LEE>LDE, inverte operacoes
            if (callbe + putb + precob) > (calla + putae + precoae):
                print("ENCERRADO - compra call, vende ativo e put")
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": call,
                    "volume": float(lot),
                    "type": mt5.ORDER_TYPE_BUY,
                    "price": calla,
                    "deviation": deviation,
                    "magic": 123456,
                    "comment": "compra",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }
                result = mt5.order_send(request)
                print(result)
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": put,
                    "volume": float(lot),
                    "type": mt5.ORDER_TYPE_SELL,
                    "price": putb,
                    "deviation": deviation,
                    "magic": 123456,
                    "comment": "venda",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }
                result = mt5.order_send(request)
                print(result)
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": ativo,
                    "volume": float(lot),
                    "type": mt5.ORDER_TYPE_SELL,
                    "price": precob,
                    "deviation": deviation,
                    "magic": 123456,
                    "comment": "venda",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }
                result = mt5.order_send(request)
                print(result)
                print("compra call, vende ativo e put")
                stop = 1
            else:
                print("AINDA ABERTA - nada a fazer")
                stop = 0
                tentativaout = tentativaout + 1
                print(tentativaout)
        if abre < 0:  # aberto LEE<LDE, inverte operacoes
            if (putbe + precobe + callb) > (callae + puta + precoa):
                print("ENCERRADO - vende call, compra ativo e put")
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": call,
                    "volume": float(lot),
                    "type": mt5.ORDER_TYPE_SELL,
                    "price": callb,
                    "deviation": deviation,
                    "magic": 123456,
                    "comment": "venda",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }
                result = mt5.order_send(request)
                print(result)
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": put,
                    "volume": float(lot),
                    "type": mt5.ORDER_TYPE_BUY,
                    "price": puta,
                    "deviation": deviation,
                    "magic": 123456,
                    "comment": "compra",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }
                result = mt5.order_send(request)
                print(result)
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": ativo,
                    "volume": float(lot),
                    "type": mt5.ORDER_TYPE_BUY,
                    "price": precoa,
                    "deviation": deviation,
                    "magic": 123456,
                    "comment": "compra",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }
                result = mt5.order_send(request)
                print(result)
                stop = 1
            else:
                print("AINDA ABERTA - nada a fazer")
                stop = 0
                tentativaout = tentativaout + 1
                print(tentativaout)


# Exemplo
putcall('PETR4','PETRI419', 'PETRU419', 38.17)
