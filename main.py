import os
from src.data.data_loading import load_data
from src.features.build_features import  engineer_features
from src.models.train_model import train_linear_regression, train_decision_tree, train_random_forest
from src.models.evaluate_model import evaluate_model, plot_decision_tree
from src.models.predict_model import make_prediction
from src.utils.utils import setup_logging
from sklearn.model_selection import train_test_split

import pickle

def main():
    logger = setup_logging()
    try:
    
        data = load_data('data/final.csv')
        logger.info("Data loaded")
        
        x, y = engineer_features(data)
        logger.info("Feature Engineering Completed")
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
        logger.info("Dataset Splitted into test and train set")
        
        lr_model = train_linear_regression(x_train, y_train)
        print(f'Linear Regression Train MAE: {evaluate_model(lr_model, x_train, y_train)}')
        print(f'Linear Regression Test MAE: {evaluate_model(lr_model, x_test, y_test)}')
        logger.info("Training Completed for Linear Regression")
        
        dt_model = train_decision_tree(x_train, y_train)        
        print(f'Decision Tree Train MAE: {evaluate_model(dt_model, x_train, y_train)}')
        print(f'Decision Tree Test MAE: {evaluate_model(dt_model, x_test, y_test)}')
        logger.info("Training Completed for Decision Tree model")
        
        plot_decision_tree(dt_model)
        
        rf_model = train_random_forest(x_train, y_train)
        print(f'Random Forest Train MAE: {evaluate_model(rf_model, x_train, y_train)}')
        print(f'Random Forest Test MAE: {evaluate_model(rf_model, x_test, y_test)}')
        logger.info("Training Completed for Random Forest model")
        
        logger.info("Saving Random Forest model")

        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'random_forest_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(rf_model, f)
        
        logger.info("Loading Random Forest model")
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        logger.info("Making predictions")
        sample_data = [[2012, 216, 74, 1, 1, 618, 2000, 600, 1, 0, 0, 6, 0]]
        prediction = make_prediction(loaded_model, sample_data)
        logger.info(f'Prediction: {prediction}')
        print(f'Prediction: {prediction}')
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

if __name__ == "__main__":
    main()
