from sklearn.metrics import mean_absolute_error
from src.utils.utils import setup_logging, save_image
import matplotlib.pyplot as plt
from sklearn import tree

logger = setup_logging()

def evaluate_model(model, x, y):
    
    try:
        predictions = model.predict(x)
        mae = mean_absolute_error(y, predictions)
        return mae
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def plot_decision_tree(dtmodel):
    try:
        # Get the features
        feature_names = list(dtmodel.feature_names_in_)

        # Plot the tree with feature names
        tree.plot_tree(dtmodel, feature_names=feature_names)

        figure = plt.gcf()  # Get the current figure
        save_image(figure, 'plot_decision_tree.png')
        
        plt.show()

    except Exception as e:
        logger.error(f"Error in Evaluate_model: {e}")
        raise e