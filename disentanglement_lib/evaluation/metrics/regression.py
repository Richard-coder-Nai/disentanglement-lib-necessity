# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of multiple classification accuracy. Includes linear regression with strong regularization, e.g., Lasso and Ridge."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
from disentanglement_lib.evaluation.metrics import utils
import numpy as np
import scipy
from six.moves import range
from sklearn import linear_model, cluster
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
import gin.tf

@gin.configurable(
    "continous_linear_regression",
    blacklist=["ground_truth_data", "representation_function", "random_state",
                "artifact_dir"])
def compute_continous_linear_regression(ground_truth_data, representation_function, random_state,
                artifact_dir=None,
                num_train=gin.REQUIRED,
                num_test=gin.REQUIRED,
                batch_size=16):
    """Computes the linear regression scores.
    Args:
        ground_truth_data: GroundTruthData to be sampled from.
        representation_function: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
        random_state: Numpy random state used for randomness.
        artifact_dir: Optional path to directory where artifacts can be saved.
        num_train: Number of points used for training.
        num_test: Number of points used for testing.
        batch_size: Batch size for sampling.
    Returns:
        Dictionary with nrmse for training and testing for linear regression.
    """
    del artifact_dir
    logging.info("Generating training set.")
    # mus_train are of shape [num_codes, num_train], while ys_train are of shape
    # [num_factors, num_train].
    mus_train, ys_train = utils.generate_batch_factor_code(
        ground_truth_data, representation_function, num_train, random_state,
        batch_size)
    logging.info("Generating testing set.")
    mus_test, ys_test = utils.generate_batch_factor_code(
        ground_truth_data, representation_function, num_test, random_state,
        batch_size)
    scores = _compute_continous_linear_regression(mus_train, ys_train, mus_test, ys_test)
    return scores

def _compute_continous_linear_regression(mus_train, ys_train, mus_test, ys_test):
    """Computes the linear regression scores.
    Args:
        mus_train: Training set representations.
        ys_train: Training set targets.
        mus_test: Test set representations.
        ys_test: Test set targets.
    Returns:
        Dictionary with continous linear regression scores.
    """
    # Compute linear regression.
    num_factors = ys_train.shape[0]
    num_codes = mus_train.shape[0]
    scores = {}
    train_nrmse = []
    test_nrmse = []
    for i in range(num_factors):
        model = linear_model.LinearRegression()
        model.fit(mus_train.T, ys_train[i, :])
        train_nrmse.append(
            _normalized_root_mse(ys_train[i, :], model.predict(mus_train.T)))
        test_nrmse.append(
            _normalized_root_mse(ys_test[i, :], model.predict(mus_test.T)))
        scores[f"train_nrmse_{i}"] = train_nrmse[-1]
        scores[f"test_nrmse_{i}"] = test_nrmse[-1]
    scores["train_nrmse"] = np.mean(train_nrmse)
    scores["test_nrmse"] = np.mean(test_nrmse)
    logging.info("Train NRMSE: %.2g", scores["train_nrmse"])
    logging.info("Test NRMSE: %.2g", scores["test_nrmse"])
    return scores

def _normalized_root_mse(target, prediction):
    """Computes normalized root mean squared error."""
    return np.sqrt(np.mean((target - prediction)**2)) / (np.max(target) - np.min(target))