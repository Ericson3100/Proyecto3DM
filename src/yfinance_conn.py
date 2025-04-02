import yfinance as yf

# Definir el símbolo del ticker para el S&P 500
ticker_symbol = "^GSPC"

historical_data = yf.download(
    tickers=ticker_symbol,
    period="max",
    interval="1d",
    auto_adjust=True,
    prepost=False,
    threads=True
)

# # Mostrar las primeras y últimas filas de los datos descargados
# print("Primeras filas de los datos históricos del S&P 500:")
# print(historical_data.head())

# print("\nÚltimas filas de los datos históricos del S&P 500:")
# print(historical_data.tail())

print(historical_data.info())
print(historical_data.describe())
print(historical_data.isnull().sum())