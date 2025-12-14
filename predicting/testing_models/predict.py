# get an input data with right specifications and predict the output
import xgboost as xgb

def predict_number_of_geese_tree(X):
    """
    Predict the actual amount of birds given input data X that should fit the tree 
    structure already
    """

    # load model
    model_filename = "predicting/models/boosted_trees/n_geese_from_3_geese.json"

    bst = xgb.Booster()
    bst.load_model(model_filename)

    # make prediction
    y_pred = bst.predict(X)

    return y_pred


