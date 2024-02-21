import logging

def logger_tessilator(name_log):
    logger = logging.getLogger(__name__)
    f_handler = logging.FileHandler(f'{name_log}.log')
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    return logger

