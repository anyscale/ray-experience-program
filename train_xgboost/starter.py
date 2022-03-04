import pandas as pd
import time

from xgboost import DMatrix, train

# Define XGBoost training parameters
xgboost_params = {
    "tree_method": "approx",
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "error"],
}


def train_xgboost(
    config: dict, data: pd.DataFrame, target_column: str, test_fraction: float = 0.3
):
    """
    Train an XGBoost model on the data specified in the `data` arg.

    Args:
        config (Dict): A dictionary for XGBoost specific configurations.
        data: (pd.DataFrame): A pandas dataframe containing the data to train on.
        target_column (str): The name of the column in the dataframe to use as the labels.
        test_fraction (float): What fraction of the data to use as the test set. 
            The test set data will be used only for evaluation and not for training. 
    """

    # Get the current time. We will use this later to see how long training takes.
    start_time = time.time()

    # Split data into train and test.
    # First sample 1-test_fraction of the data to use as the training set.
    X_train = data.sample(frac=1 - test_fraction)
    # Then take the remainder of the data and use that as the test set.
    X_test = data.drop(X_train.index)

    # Pass Pandas dataframe to DMatrix. 
    train_set = DMatrix(X_train.drop(target_column, axis=1), X_train[target_column])
    test_set = DMatrix(X_test.drop(target_column, axis=1), X_test[target_column])

    evals_result = {}

    # Run the training
    bst = train(
        params=config,
        dtrain=train_set,
        evals=[(test_set, "eval")],
        evals_result=evals_result,
        verbose_eval=False,
        num_boost_round=10,
    )
    print(f"Total time taken: {time.time()-start_time}")

    # Get a model back
    model_path = "model.xgb"
    bst.save_model(model_path)
    print("Final validation error: {:.4f}".format(evals_result["eval"]["error"][-1]))

    return bst

# Use pandas to read the parquet files in the ./data directory to a pandas dataframe.
data = pd.read_parquet(
    "./data"
)
bst = train_xgboost(
    xgboost_params,
    data,
    "label",
)
