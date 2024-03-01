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

"""Implementation of linear algebra metrics.
E.g., eigengaps.

Inspired by dimensional collapse.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
from disentanglement_lib.evaluation.metrics import utils
import numpy as np
import scipy
from six.moves import range
from sklearn import ensemble, linear_model, preprocessing
import gin.tf
import tensorflow.compat.v1 as tf
from matplotlib import pyplot as plt
import seaborn as sns


@gin.configurable(
    "eigen_gaps",
    blacklist=["ground_truth_data", "representation_function", "random_state", ])
def compute_eigen(ground_truth_data, representation_function, random_state,
                artifact_dir=None,
                num_train=gin.REQUIRED,
                batch_size=16, 
                log_thresh=-5):
  """Computes the eigen gap scores.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.
    log_thresh: Threshold of inactive dimension, values smaller will be considered inactive.

  Returns:
    Dictionary with eigen gaps.
  """
  logging.info("Generating training set.")
  # mus_train are of shape [num_codes, num_train], while ys_train are of shape
  # [num_factors, num_train].
  mus_train, ys_train = utils.generate_batch_factor_code(
      ground_truth_data, representation_function, num_train,
      random_state, batch_size)
  assert mus_train.shape[1] == num_train
  assert ys_train.shape[1] == num_train
  scores = _compute_eigen(mus_train, log_thresh,  artifact_dir)
  return scores

def _compute_eigen(mus, log_thresh, artifact_dir):
  """Computes eigengaps of cov matrix."""
  scores = {}
  cov = np.cov(mus)
  scores["log condition number"] = np.log10(np.linalg.cond(cov))
  logging.info(f"Condition number is {scores['log condition number']}.")
  # Number of sigular values before largest gap.
  v = scipy.linalg.svdvals(cov)
  logv = np.log10(v)
  gap_idx = np.abs(np.diff(logv)).argmax()
  thresh_idx = np.searchsorted(-logv, -log_thresh)
  scores["active dim score"] = min(thresh_idx, (gap_idx + 1)) / len(v)
  logging.info(f"Gap index is {gap_idx}")
  logging.info(f"Thresh index is {thresh_idx}")
  logging.info(f"Active dimension socre is {scores['active dim score']}")
  sns.set()
  fig_plt = plt.figure()
  ax = fig_plt.add_subplot()
  plot = sns.lineplot(x=np.arange(len(v)), y=logv, ax=ax)
  plot.set_xlabel("Singular Value Index")
  plot.set_ylabel("Log of singular values")
  fig_plt.savefig(f'{artifact_dir}/log_singular.png', bbox_inches='tight')
  return scores

