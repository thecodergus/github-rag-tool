import logging

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configura e retorna um logger com stream handler e formatter padrão.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger