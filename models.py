#!/usr/bin/env python

import lightgbm as lgb
import numpy as np


def test_lgb(Xtrain, ytrain, Xtest, ytest, model_type='test',num_estimators=500,return_preds=False, return_model=False):
    # Xtrain, ytrain, Xtest, ytest = split_train_test(db)
    # model = joblib.load(model_params_path)
    # model.set_params(objective='regression', metric='rmse')

    if model_type == 'overfit':
        model = lgb.LGBMRegressor(
            learning_rate=0.5,       # Even more aggressive learning
            max_depth=-1,            # No limit on depth
            num_leaves=2**16,        # A massive number of leaves to force splits
            min_child_samples=1,     # Allow splitting until only 1 sample remains
            n_estimators=5000,       # Huge number of trees to ensure memorization
            reg_alpha=0,             # No L1 regularization
            reg_lambda=0,            # No L2 regularization
            subsample=1.0,           # Use 100% of the data
            colsample_bytree=1.0,    # Use 100% of features
            min_split_gain=0,        # Allow splits even with **no gain**
            importance_type='gain',  # Prioritize gain-based splitting
            force_col_wise=True,     # Helps force more splits
            extra_trees=True,        # Randomizes split selection to allow more splits
            deterministic=True,      # Ensures no randomness
            boosting_type='gbdt',    # Standard boosting
            objective='regression',
            metric='rmse',
            random_state=42,
            n_jobs=1    # <------- setting `n_jobs` prevents segfaults when running LightGBM from inside a script 
        )

    elif model_type == 'paper':
        model = lgb.LGBMRegressor(
        learning_rate=0.15, 
        max_depth=9,
        # min_child_samples=5,
        n_estimators=39,
        num_leaves = 35,
        random_state=399,
        # reg_alpha=0.2,
        # reg_lambda=0.01,
        objective='regression',
        metric='rmse',
        n_jobs=1)

    else:
        model = lgb.LGBMRegressor(
        learning_rate=0.05, 
        max_depth=-1,
        min_child_samples=5,
        n_estimators=num_estimators,
        num_leaves = 2**12,
        random_state=399,
        reg_alpha=0.2,
        reg_lambda=0.01,
        objective='regression',
        metric='rmse',
        n_jobs=1)

    model.fit(Xtrain,ytrain)
    ytest_pred = model.predict(Xtest) 
    r = np.corrcoef(ytest, ytest_pred)[0,1]

    return_bools = [return_preds, return_model]

    if any(return_bools):
        extra_outputs = [ytest_pred, model]
        return r, *[out for (out, retbool) in zip(extra_outputs, return_bools) if retbool]
    else:
        return r