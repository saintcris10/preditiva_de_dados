
import yfinance as yf
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from datetime import date

lista_ticker = ["PETR4.SA", "ABEVE.SA","MGLU3.SA","BBAS3.SA","GOOG","AAPL","MSFT"]


tickers = {
"PETR4.SA": "Petrobras",
"ABEVE.SA": "Ambev",
"MGLU3.SA": "Magazine Luiza",
"BBAS3.SA": "Banco do Brasil",
"GOOG": "Google",
"AAPL": "Apple",
"MSFT": "Microsoft"
}

def carregar_dados(ticker,dt_inicial,dt_final):
    df = yf.Ticker(ticker).history(start= dt_inicial.strftime("%Y-%m-%d"), 
                                   end=dt_final.strftime("%Y-%m-%d"))
    
    return df

def prever_dados(df,periodo):
    df.reset_index(inplace=True)
    df = df.loc[:, ['Date','Close']]
    df['Date'] = df['Date'].dt.tz_localize(None)
    df.rename(columns= {"Date": "ds", "Close": "y"}, inplace=True)

    modelo= Prophet()
    modelo.fit(df)

    datas_futuras = modelo.make_future_dataframe(periods=int(periodo) * 30)
    previsoes = modelo.predict(datas_futuras)
    return modelo, previsoes


st.image("logo.jpg")
st.markdown("""
# Analises Preditivas
### Prevendo Valores da Bolsa de Valores           
""")

with st.sidebar:
    st.header("menu")
    ticker = st.selectbox("selecione uma ação:",lista_ticker)
    dt_inicial = st.date_input("Data inicial", value= date(2020,1,1))
    dt_final = st.date_input("Data final")
    meses = st.number_input("Meses de Previsão", 1, 24, value=6)

dados = carregar_dados(ticker, dt_inicial, dt_final)

if dados.shape[0] != 0:

    st.header(f"dados da ação - {tickers[ticker]}")
    st.dataframe(dados)
    st.subheader("Variação do periodo")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dados.index, y=dados['Close'], name="Open" ))
    st.plotly_chart(fig)

    st.header(f"Previsão proximo(s) {meses} meses")
    modelo,previsoes = prever_dados(dados,meses)
    fig = plot_plotly(modelo, previsoes, xlabel="período", ylabel="valor")
    st.plotly_chart(fig)

else:
    st.warning("Nenhum dado encomtrado no periodo selecionado")