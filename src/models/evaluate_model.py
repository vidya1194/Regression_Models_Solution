from sklearn.metrics import mean_absolute_error
from src.utils.utils import setup_logging

def evaluate_model(model, x, y):
    logger = setup_logging()
    try:
        predictions = model.predict(x)
        mae = mean_absolute_error(y, predictions)
        return mae
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
