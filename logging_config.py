import logging

# from rich.logging import RichHandler


def get_logger(name="change-predictor", level=logging.INFO, model_name: str = None, simil_cutoff: float = None):
    logger = logging.getLogger(name)

    log_filename = "application"

    if not logger.hasHandlers():
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(message)s",
            # handlers=[RichHandler(markup=True)]
            filename=f"{log_filename}.log",
        )
        logger.info(f"{name} Logger has been initialized...")
    return logger