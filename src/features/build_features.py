from src.utils.utils import setup_logging


def engineer_features(data):
    """
    Engineer features from the dataset.
    """
    logger = setup_logging()
    try:
        features = data.drop('price', axis=1)
        target = data['price']
        return features, target
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
