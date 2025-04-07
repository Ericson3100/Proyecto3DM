import yfinance as yf
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import os


def download_stock_data(ticker_symbols, period="max", interval="1d", end_date="2025-02-28", save_csv=True):
    """
    Descarga datos históricos de acciones usando yfinance y filtra hasta una fecha específica.
    
    Args:
        ticker_symbols (str or list): Símbolo(s) del ticker a descargar. Puede ser un string para un solo ticker
                                    o una lista para múltiples tickers.
        period (str): Período de tiempo a descargar (por defecto: "max" para obtener todo el histórico disponible)
        interval (str): Intervalo de tiempo entre datos (por defecto: "1d" para diario)
        end_date (str): Fecha final para filtrar los datos en formato 'YYYY-MM-DD' (por defecto: "2025-02-28")
        save_csv (bool): Si es True, guarda los datos en archivos CSV (por defecto: True)
        
    Returns:
        dict: Diccionario con DataFrames para cada ticker, cada uno con sus indicadores técnicos
    """
    # Convertir ticker_symbols a lista si es un string
    if isinstance(ticker_symbols, str):
        ticker_symbols = [ticker_symbols]
    
    # Diccionario para almacenar los DataFrames de cada ticker
    ticker_data = {}
    
    # Crear la carpeta ./data/raw si no existe y si se van a guardar CSV
    if save_csv:
        os.makedirs("./data/raw", exist_ok=True)
    
    for ticker in ticker_symbols:
        # Usar yfinance para descargar los datos históricos
        historical_data = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            prepost=False,
            threads=True
        )
        
        # Asegurar que es un DataFrame y resetear el índice para tener la fecha como columna
        historical_data = pd.DataFrame(historical_data).reset_index()
        
        # Renombrar las columnas para simplificar
        historical_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Filtrar los datos hasta la fecha final especificada
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        historical_data = historical_data[historical_data['Date'] <= end_date_obj]
        
        print(f"Shape del DataFrame para {ticker}: {historical_data.shape}")
        print(f"Rango de fechas para {ticker}: {historical_data['Date'].min()} a {historical_data['Date'].max()}")
        
        # Añadir indicadores técnicos
        # SMA de 20 y 50 días
        historical_data['SMA_20'] = ta.sma(historical_data['Close'], length=20)
        historical_data['SMA_50'] = ta.sma(historical_data['Close'], length=50)
        
        # RSI de 14 períodos
        historical_data['RSI_14'] = ta.rsi(historical_data['Close'], length=14)
        
        # MACD
        macd = ta.macd(historical_data['Close'])
        historical_data = pd.concat([historical_data, macd], axis=1)
        
        # Bandas de Bollinger
        bollinger = ta.bbands(historical_data['Close'], length=20)
        historical_data = pd.concat([historical_data, bollinger], axis=1)
        
        # Almacenar el DataFrame en el diccionario
        ticker_data[ticker] = historical_data
        
        # Guardar el DataFrame en un archivo CSV si save_csv es True
        if save_csv:
            file_path = f"./data/raw/{ticker}_data.csv"
            historical_data.to_csv(file_path, index=False)
            print(f"Datos guardados en {file_path}")
        
        print(f"Indicadores calculados para {ticker}:", 
              [col for col in historical_data.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']])
    
    return ticker_data

# Ejemplo de uso: descargar datos de múltiples tickers
tickers = ["^GSPC", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]  # S&P 500, Apple, Microsoft
ticker_data = download_stock_data(tickers, period="max", interval="1d", end_date="2025-02-28", save_csv=False)

# Añadir datos del S&P 500 a cada acción individual
sp500_data = ticker_data["^GSPC"]
for ticker in tickers:
    if ticker != "^GSPC":  # Evitar añadir S&P 500 a sí mismo
        # Combinar datos de la acción con S&P 500 basado en la fecha
        stock_data = ticker_data[ticker]
        merged_data = pd.merge(stock_data, 
                              sp500_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']], 
                              on='Date', 
                              how='left',
                              suffixes=('', '_SP500'))
        
        # Actualizar el DataFrame en el diccionario
        ticker_data[ticker] = merged_data

        file_path = f"./data/raw/{ticker}_with_SP500.csv"
        merged_data.to_csv(file_path, index=False)
        print(f"Datos con S&P 500 guardados en {file_path}")
