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

"""Implementation of the pair predict score.

1. We sample a batch of paired data. Each pair is labled with a 
num_factors-d binary vector, indicating each factor is equal or not.
2. Then we train a linear classifier to predict the labels from the 
concatenate of representations of each pair.
Sampling code based on beta-VAE score.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
import numpy as np
from six.moves import range
from sklearn import linear_model
import gin.tf


@gin.configurable(
    "pair_predict",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compute_pair_predict(ground_truth_data,
                             representation_function,
                             random_state,
                             artifact_dir=None,
                             num_train=gin.REQUIRED,
                             num_eval=gin.REQUIRED):
  """Computes the pair predict score metric using scikit-learn.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    num_eval: Number of points used for evaluation.

  Returns:
    Dictionary with scores:
      train_accuracy: Accuracy on training set.
      eval_accuracy: Accuracy on evaluation set.
  """
  del artifact_dir
  logging.info("Generating training set.")
  train_points, train_labels = _generate_training_batch(
      ground_truth_data, representation_function, num_train,
      random_state)

  logging.info("Generating evaluation set.")
  eval_points, eval_labels = _generate_training_batch(
      ground_truth_data, representation_function,  num_eval,
      random_state)

  scores_dict = {}
  train_acc = []
  eval_acc = []
  num_factors = train_labels.shape[1]
  for i in range(num_factors):
    model = linear_model.RidgeClassifier(alpha=10.0, random_state=random_state)
    model.fit(train_points, train_labels[:,i])
    train_acc.append(model.score(train_points, train_labels[:,i]))
    eval_acc.append(model.score(eval_points, eval_labels[:,i]))
    scores_dict[f'train_accuracy{i}'] = train_acc[-1]
    scores_dict[f'eval_accuracy{i}'] = eval_acc[-1]
    
  train_accuracy = np.mean(train_acc)
  eval_accuracy = np.mean(eval_acc)
  
  logging.info("Training set accuracy: %.2g", train_accuracy)
  logging.info("Evaluation set accuracy: %.2g", eval_accuracy)
  scores_dict["train_accuracy"] = train_accuracy
  scores_dict["eval_accuracy"] = eval_accuracy
  return scores_dict


def _generate_training_batch(ground_truth_data, representation_function,
                             num_points, random_state):
  """Sample a set of training samples based on a batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    num_points: Number of points to be sampled for training set.
    random_state: Numpy random state used for randomness.

  Returns:
    points: (num_points, dim_representation)-sized numpy array with training set
      features.
    labels: (num_points)-sized numpy array with training set labels.
  """
  num_factors = ground_truth_data.num_factors
  batch_size = num_points // num_factors
  points = []
  labels = []
  
  for i in range(num_factors):
    label, feature_vector = _generate_training_sample(
        ground_truth_data, representation_function, i, batch_size, random_state)
    labels.append(label)
    points.append(feature_vector)
  
  points = np.concatenate(points, axis=0)
  labels = np.concatenate(labels, axis=0)
  idx = random_state.permutation(len(points))
  points, labels = points[idx], labels[idx]
  
  return points, labels


def _generate_training_sample(ground_truth_data, representation_function, index, 
                              batch_size, random_state):
  """Sample a single training sample based on a mini-batch of ground-truth data.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Number of points to be used to compute the training_sample
    index: Index of factors fixed.
    random_state: Numpy random state used for randomness.

  Returns:
    feature_vector: Feature vector of training sample.
    label: #factor-d binary vecotors.
  """
  # Sample two mini batches of latent variables.
  factors1 = ground_truth_data.sample_factors(batch_size, random_state)
  factors2 = ground_truth_data.sample_factors(batch_size, random_state)
  # Ensure sampled coordinate is the same across pairs of samples.
  factors2[:, index] = factors1[:, index]
  # Transform latent variables to observation space.
  observation1 = ground_truth_data.sample_observations_from_factors(
      factors1, random_state)
  observation2 = ground_truth_data.sample_observations_from_factors(
      factors2, random_state)
  # Compute representations based on the observations.
  representation1 = representation_function(observation1)
  representation2 = representation_function(observation2)
  # Compute the feature vector based on contatenation of representation.
  feature_vector = np.concatenate([representation1, representation2], axis=1)
  # feature_vector = np.abs(representation1 - representation2)
  label = (factors1==factors2).astype(np.int)
  return label, feature_vector
