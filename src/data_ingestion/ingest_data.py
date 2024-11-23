import yfinance as yf
import pandas as pd
import pandas_market_calendars as mcal
import os
from datetime import datetime, timedelta
import logging
from src.config import setup_logging

# Configurar o logging
setup_logging()
logger = logging.getLogger(__name__)


def load_existing_data(file_path: str) -> pd.DataFrame:
    """
    Carrega os dados processados existentes.

    Args:
        file_path: Caminho para o arquivo CSV.
    Returns:
        DataFrame com os dados existentes
    """
    if os.path.exists(file_path):
        logger.info(f"Carregando dados existentes de {file_path}")
        return pd.read_csv(file_path, parse_dates=['Date'])

    logger.warning(
        f"Arquivo {file_path} nao encontrado. Criando novo DataFrame.")
    return pd.DataFrame()


def save_data(
    df: pd.DataFrame,
    file_path: str,
    format: str = 'csv'
) -> None:
    """
    Salva os dados processados em um arquivo.

    Args:
        df: DataFrame com os dados processados.
        file_path: Caminho para o arquivo CSV de saida.
    Returns:
        None
    """
    file_format = file_path.split('.')[-1]
    if format != file_format:
        raise ValueError(
            f"""Formato de saida [{format}] diferente do
             formato do arquivo [{file_format}]""")
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    logger.info(f"Salvando dados em {file_path}")
    if format == 'parquet':
        df.to_parquet(file_path)
    elif format == 'csv':
        df.to_csv(file_path, index=False)
    else:
        raise ValueError(f"Formato {format} nao suportado.")


def get_last_date(df: pd.DataFrame) -> datetime:
    """
    Obtém a última data presente no DataFrame.

    Args:
        df: DataFrame com os dados existentes.
    Returns:
        Última data presente no DataFrame.
    """
    if not df.empty:
        last_date = df['Date'].max()
        logger.info(f"Última data existente: {last_date}")
        return last_date

    logger.info("DataFrame vazio. Não há última data.")
    return None


def download_new_data(
    symbol: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Baixa novos dados de acoes do Yahoo Finance.

    Args:
        symbol: Simbolo da acao.
        start_date: Data de inicio.
        end_date: Data final.
    Returns:
        DataFrame com os novos dados.
    """
    logger.info(f"Baixando dados de {symbol} de {start_date} ate {end_date}")
    return yf.download(symbol, start=start_date, end=end_date).reset_index()


def append_new_data(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Appenda novos dados ao DataFrame existente e salva no CSV.

    Args:
        existing_df: DataFrame com os dados existentes.
        new_df: DataFrame com os novos dados.
        output_path: Caminho para o arquivo CSV de saída.
    Returns:
        None
    """
    # Resetar o índice para transformar 'Date' em coluna
    existing_df = existing_df.reset_index()
    new_df = new_df.reset_index()

    if existing_df.empty:
        combined_df = new_df
        logger.info("DataFrame existente vazio. Utilizando apenas os novos dados.")
    else:
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(
            subset=['Date']).sort_values('Date')
        logger.info("Novos dados adicionados ao DataFrame existente.")

    # Opcionalmente, redefinir 'Date' como índice após remover duplicatas
    # combined_df.set_index('Date', inplace=True)

    # Salvar os dados
    save_data(combined_df, output_path)
    logger.info(f"Dados atualizados salvos em {output_path}")


def validate_data_continuity(df: pd.DataFrame):
    """
    Valida a continuidade dos dados de acordo com o calendário de mercado.

    :param df: DataFrame com os dados existentes.
    """
    # Obter o calendário da B3 (Bolsa de Valores do Brasil)
    b3 = mcal.get_calendar('BVMF')
    schedule = b3.schedule(start_date=df['Date'].min(), end_date=df['Date'].max())
    all_days = schedule.index  # Datas em que o mercado esteve aberto

    existing_days = pd.to_datetime(df['Date'])
    missing_days = set(all_days) - set(existing_days)
    if len(missing_days) == 0:
        logger.info("Validação de continuidade dos dados bem-sucedida.")
    else:
        logger.warning(
            f"Faltam dados para as seguintes datas (mercado aberto): {missing_days}")


def main():

    SYMBOL = 'VIVT3.SA'
    PROCESSED_DATA_PATH = 'data/processed/vivo_processed.csv'

    # Carregar dados existentes
    existing_df = load_existing_data(PROCESSED_DATA_PATH)

    if not existing_df.empty:
        last_date = get_last_date(existing_df)
        # Definir o próximo dia útil após a última data
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        # Se não houver dados, começar de uma data inicial
        start_date = '2018-01-01'

    # Data final será a data atual
    end_date = datetime.today().strftime('%Y-%m-%d')

    logger.info(f"Baixando dados de {start_date} até {end_date}")

    # Baixar novos dados
    new_df = download_new_data(SYMBOL, start_date, end_date)

    if new_df.empty:
        logger.info("Nenhum novo dado para adicionar.")
        return

    # Appendar novos dados
    append_new_data(existing_df, new_df, PROCESSED_DATA_PATH)

    # Recarregar os dados combinados para validação
    combined_df = load_existing_data(PROCESSED_DATA_PATH)

    # Validar a continuidade dos dados
    try:
        validate_data_continuity(combined_df)
    except AssertionError as e:
        logger.error(str(e))


if __name__ == "__main__":
    main()
