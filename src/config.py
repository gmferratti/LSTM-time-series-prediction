import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(log_level=logging.INFO, log_file='logs/app.log'):
    """
    Configura o sistema de logging.

    Args:
        log_level: Nivel de logging desejado.
        log_file: Caminho para o arquivo de log.
    """
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    # Formato do log
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configurar o logger raiz
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Handler para o console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # Handler para o arquivo com rotação
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5*1024*1024, backupCount=5)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
