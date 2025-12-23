# get an input data with right specifications and predict the output
import xgboost as xgb


def predict_number_of_geese_small_tree(X):
    """
    Predict the actual amount of birds given input data X that should be np array.
    """

    # load model
    model_filename = "/home/louis/geese_project/predicting/models/boosted_trees/n_geese_from_3_geese_small.json"

    bst = xgb.Booster()
    bst.load_model(model_filename)

    # make prediction
    y_pred = bst.predict(X)

    # Get a text dump of the trees
    trees_dump = bst.get_dump()

    # Each node/leaf in the dump starts with a digit followed by a colon (e.g., "0:[f5<...]")
    total_nodes = sum(tree.count(":") for tree in trees_dump)

    return y_pred, total_nodes


def predict_number_of_geese_big_tree(X):
    """
    Predict the actual amount of birds given input data X that should be np array.
    """

    # load model
    model_filename = "/home/louis/geese_project/predicting/models/boosted_trees/n_geese_from_3_geese_big.json"

    bst = xgb.Booster()
    bst.load_model(model_filename)

    # make prediction
    y_pred = bst.predict(X)

    trees_dump = bst.get_dump()
    total_nodes = sum(tree.count(":") for tree in trees_dump)

    return y_pred, total_nodes
