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

"""Implementation of KNN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
from disentanglement_lib.evaluation.metrics import utils
import numpy as np
import scipy
from six.moves import range
from sklearn import neighbors
import gin.tf

@gin.configurable(
    "neighbors",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compurte_neighbors(ground_truth_data, representation_function, random_state,
                artifact_dir=None,
                num_train=gin.REQUIRED,
                num_test=gin.REQUIRED,
                batch_size=16):
  """Computes the neighbors classification scores.
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
    Dictionary with accracy for KNN classifier.
  """
  del artifact_dir
  logging.info("Generating training set.")
  # mus_train are of shape [num_codes, num_train], while ys_train are of shape
  # [num_factors, num_train].
  mus_train, ys_train = utils.generate_batch_factor_code(
      ground_truth_data, representation_function, num_train,
      random_state, batch_size)
  assert mus_train.shape[1] == num_train
  assert ys_train.shape[1] == num_train
  mus_test, ys_test = utils.generate_batch_factor_code(
      ground_truth_data, representation_function, num_test,
      random_state, batch_size)
  scores = _compute_knn(mus_train, ys_train, mus_test, ys_test)
  return scores
    
@gin.configurable("knn", 
        blacklist=["mus_train", "ys_train", "mus_test", "ys_test"])
def _compute_knn(mus_train, ys_train, mus_test, ys_test):
  """Computes knn classification accuracy."""
  num_factors = ys_train.shape[0]
  num_codes = mus_train.shape[0]
  scores = {}
  train_acc = []
  test_acc = []
  for i in range(num_factors):
    model = neighbors.KNeighborsClassifier()
    model.fit(mus_train.T, ys_train[i, :])
    train_acc.append(np.mean(model.predict(mus_train.T) == ys_train[i, :]))
    test_acc.append(np.mean(model.predict(mus_test.T) == ys_test[i, :]))
    scores[f"knn_accuracy_train{i}"] = train_acc[-1]
    scores[f"knn_accuracy_test{i}"] = test_acc[-1]
  scores["knn_accuracy_train"] = np.mean(train_acc)
  scores["knn_accuracy_test"] = np.mean(test_acc)
  logging.info("Training set knn accuracy: %.2g", scores["knn_accuracy_train"])
  logging.info("Evaluation set knn accuracy: %.2g", scores["knn_accuracy_test"])
  return scores
