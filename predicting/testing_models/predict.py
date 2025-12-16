# get an input data with right specifications and predict the output
import xgboost as xgb


def predict_number_of_geese_tree(X):
    """
    Predict the actual amount of birds given input data X that should be np array.
    """

    # load model
    model_filename = "/home/louis/geese_project/predicting/models/boosted_trees/n_geese_from_3_geese_small.json"

    bst = xgb.Booster()
    bst.load_model(model_filename)

    # make prediction
    y_pred = bst.predict(X)

    return y_pred
