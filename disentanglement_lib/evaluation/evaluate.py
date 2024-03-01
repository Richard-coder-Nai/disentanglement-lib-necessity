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

"""Evaluation protocol to compute metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import os
import time
import warnings
import contextlib
import numpy as np

from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.evaluation.metrics import beta_vae  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import pair_predict  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import dci  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import med  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import linalg  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import classification  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import neighbors  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import downstream_task  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import factor_vae  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import fairness  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import irs  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import mig  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import modularity_explicitness  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import reduced_downstream_task  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import sap_score  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import strong_downstream_task  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import unified_scores  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import unsupervised_metrics  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import regression  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.utils import results
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

import gin.tf

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def get_rotation_mat_QR(z_dim, rotate_seed):
    """Generate an orthonomal matrix by QR factorization."""
    with temp_seed(rotate_seed):
        m = np.random.randn(z_dim, z_dim)
        q, _ = np.linalg.qr(m)
    return q

def evaluate_with_gin(model_dir,
                      output_dir,
                      overwrite=False,
                      gin_config_files=None,
                      gin_bindings=None):
  """Evaluate a representation based on the provided gin configuration.

  This function will set the provided gin bindings, call the evaluate()
  function and clear the gin config. Please see the evaluate() for required
  gin bindings.

  Args:
    model_dir: String with path to directory where the representation is saved.
    output_dir: String with the path where the evaluation should be saved.
    overwrite: Boolean indicating whether to overwrite output directory.
    gin_config_files: List of gin config files to load.
    gin_bindings: List of gin bindings to use.
  """
  if gin_config_files is None:
    gin_config_files = []
  if gin_bindings is None:
    gin_bindings = []
  gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
  evaluate(model_dir, output_dir, overwrite)
  gin.clear_config()


@gin.configurable(
    "evaluation", blacklist=["model_dir", "output_dir", "overwrite"])
def evaluate(model_dir,
             output_dir,
             overwrite=False,
             evaluation_fn=gin.REQUIRED,
             random_seed=gin.REQUIRED,
             name="",
             rotate_embedding=False,
             rotate_seed=0, 
             scaler=None):
  """Loads a representation TFHub module and computes disentanglement metrics.

  Args:
    model_dir: String with path to directory where the representation function
      is saved.
    output_dir: String with the path where the results should be saved.
    overwrite: Boolean indicating whether to overwrite output directory.
    evaluation_fn: Function used to evaluate the representation (see metrics/
      for examples).
    random_seed: Integer with random seed used for training.
    name: Optional string with name of the metric (can be used to name metrics).
    rotate_embedding: Whether to rotate model output.
    rotate_seed: Random seed to generate rotation matrix.
    scaler: Funtion to scale representations.
  """
  # Delete the output directory if it already exists.
  if tf.gfile.IsDirectory(output_dir):
    if overwrite:
      tf.gfile.DeleteRecursively(output_dir)
    else:
      raise ValueError("Directory already exists and overwrite is False.")

  # Set up time to keep track of elapsed time in results.
  experiment_timer = time.time()

  # Automatically set the proper data set if necessary. We replace the active
  # gin config as this will lead to a valid gin config file where the data set
  # is present.
  if gin.query_parameter("dataset.name") == "auto":
    # Obtain the dataset name from the gin config of the previous step.
    gin_config_file = os.path.join(model_dir, "results", "gin",
                                   "postprocess.gin")
    gin_dict = results.gin_dict(gin_config_file)
    with gin.unlock_config():
      gin.bind_parameter("dataset.name", gin_dict["dataset.name"].replace(
          "'", ""))
  dataset = named_data.get_named_ground_truth_data()

  # Path to TFHub module of previously trained representation.
  module_path = os.path.join(model_dir, "tfhub")
  with hub.eval_function_for_module(module_path) as f:
    # Calculate global std and mean.
    random_state = np.random.RandomState(random_seed)
    observations = dataset.sample_observations(10000, random_state=random_state)
    def _tmp_representation_function(x):
      """representation function to calculate global std and mean."""
      output = f(dict(images=x), signature="representation", as_dict=True)
      return np.array(output["default"])
    representations = utils.obtain_representation(observations,
                                                _tmp_representation_function,
                                                64)
    representations = np.transpose(representations)
    std,mean = np.std(representations, axis=0, ddof=1),np.mean(representations, axis=0)
    with gin.unlock_config():
      gin.bind_parameter("global_stat.mean", mean)
      gin.bind_parameter("global_stat.std", std)


    def _representation_function(x):
      """Computes representation vector for input images."""
      output = f(dict(images=x), signature="representation", as_dict=True)
      # Apply scaler
      if scaler is not None:
        scaler_f = scaler()
        output["default"] = scaler_f(output["default"])
      if rotate_embedding:
        embedding = output["default"]
        q = get_rotation_mat_QR(embedding.shape[-1], rotate_seed)
        return embedding @ q
      else:
        return np.array(output["default"])

    # Computes scores of the representation based on the evaluation_fn.
    if _has_kwarg_or_kwargs(evaluation_fn, "artifact_dir"):
      artifact_dir = os.path.join(output_dir, "artifacts", name)
      if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
      results_dict = evaluation_fn(
          dataset,
          _representation_function,
          random_state=np.random.RandomState(random_seed),
          artifact_dir=artifact_dir)
    else:
      # Legacy code path to allow for old evaluation metrics.
      warnings.warn(
          "Evaluation function does not appear to accept an"
          " `artifact_dir` argument. This may not be compatible with "
          "future versions.", DeprecationWarning)
      results_dict = evaluation_fn(
          dataset,
          _representation_function,
          random_state=np.random.RandomState(random_seed))

  # Save the results (and all previous results in the pipeline) on disk.
  original_results_dir = os.path.join(model_dir, "results")
  results_dir = os.path.join(output_dir, "results")
  if not os.path.exists(original_results_dir):
    original_results_dir = None
  results_dict["elapsed_time"] = time.time() - experiment_timer
  results.update_result_directory(results_dir, "evaluation", results_dict,
                                  original_results_dir)


def _has_kwarg_or_kwargs(f, kwarg):
  """Checks if the function has the provided kwarg or **kwargs."""
  # For gin wrapped functions, we need to consider the wrapped function.
  if hasattr(f, "__wrapped__"):
    f = f.__wrapped__
  (args, _, kwargs, _, _, _, _) = inspect.getfullargspec(f)
  if kwarg in args or kwargs is not None:
    return True
  return False
