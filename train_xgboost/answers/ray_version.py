import ray
import ray.data
import time

from xgboost_ray import RayDMatrix, train, RayParams

# XGBoost config.
xgboost_params = {
    "tree_method": "approx",
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "error"],
}


def train_xgboost_ray(
    config, data: ray.data.Dataset, target_column: str, ray_params: RayParams, test_fraction: float = 0.3
):
    start_time = time.time()

    split_index = int(data.count() * (1 - test_fraction))
    X = data.random_shuffle()
    X_train, X_valid = X.split_at_indices([split_index])

    ds_train = X_train.repartition(16)
    ds_valid = X_valid.repartition(16)
    train_set = RayDMatrix(ds_train, target_column, ignore=["partition"])
    test_set = RayDMatrix(ds_valid, target_column, ignore=["partition"])

    evals_result = {}

    # Train the classifier
    bst = train(
        params=config,
        dtrain=train_set,
        evals=[(test_set, "eval")],
        evals_result=evals_result,
        verbose_eval=False,
        num_boost_round=10,
        ray_params=ray_params,
    )
    print(f"Total time taken: {time.time()-start_time}")

    model_path = "model.xgb"
    bst.save_model(model_path)
    print("Final validation error: {:.4f}".format(evals_result["eval"]["error"][-1]))

    return bst


data = ray.data.read_parquet(
    "s3://anyscale-demo/data/classification.parquet/partition=0/part_0.parquet"
)
bst = train_xgboost_ray(
    xgboost_params, data, "labels", RayParams(num_actors=4, cpus_per_actor=1)
)
