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
    config, data: pd.DataFrame, target_column: str, test_fraction: float = 0.3
):
    start_time = time.time()

    # Split data into train and test
    X_train = data.sample(frac=1 - test_fraction)
    X_valid = data.drop(X_train.index)

    # Pass Pandas dataframe to DMatrix
    train_set = DMatrix(X_train.drop(target_column, axis=1), X_train[target_column])
    test_set = DMatrix(X_valid.drop(target_column, axis=1), X_valid[target_column])

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


data = pd.read_parquet(
    "./data"
)
bst = train_xgboost(
    xgboost_params,
    data,
    "label",
)
