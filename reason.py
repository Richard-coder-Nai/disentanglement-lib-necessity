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

"""Main training protocol used for computing abstract reasoning scores.

This is the main pipeline for the reasoning step in the paper:
Are Disentangled Representations Helpful for Abstract Visual Reasoning?
Sjoerd van Steenkiste, Francesco Locatello, Juergen Schmidhuber, Olivier Bachem.
NeurIPS, 2019.
https://arxiv.org/abs/1905.12506
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
import os
import sys
import time
import argparse
import reason_models as models
from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.evaluation.abstract_reasoning import pgm_data
from disentanglement_lib.utils import results
from disentanglement_lib.evaluation.metrics import utils
from absl import logging
import tensorflow_hub as hub
import numpy as np
import gin.tf.external_configurables  # pylint: disable=unused-import
import gin.tf
from tensorflow.contrib import tpu as contrib_tpu
import itertools


def _compute_std_and_mean(batch_size=10000, eval_batch_size=256):
  """Computes the variance and mean for each dimension of the representation."""
  ground_truth_data = named_data.get_named_ground_truth_data()
  random_seed = gin.query_parameter("abstract_reasoning.random_seed")
  random_state = np.random.RandomState(random_seed)
  observations = ground_truth_data.sample_observations(batch_size, random_state)
  hub_path = gin.query_parameter("HubEmbedding.hub_path")
  logging.info("Computing global variances and mean to standardise.")
  with hub.eval_function_for_module(hub_path) as f:
    def _representation_function(x):
      """Computes representation vector for input images."""
      output = f(dict(images=x), signature="representation", as_dict=True)
      return np.array(output["default"])
    representations = utils.obtain_representation(observations,
                                                _representation_function,
                                                eval_batch_size)
  representations = np.transpose(representations)
  assert representations.shape[0] == batch_size
  std,mean = np.std(representations, axis=0, ddof=1),np.mean(representations, axis=0)
  return std, mean

def reason_with_gin(input_dir,
                    output_dir,
                    overwrite=False,
                    gin_config_files=None,
                    gin_bindings=None):
  """Trains a model based on the provided gin configuration.

  This function will set the provided gin bindings, call the reason() function
  and clear the gin config. Please see reason() for required gin bindings.

  Args:
    input_dir: String with path to BYOL onnx file. 
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
  reason(input_dir, output_dir, overwrite)
  gin.clear_config()


@gin.configurable(
    "abstract_reasoning", blacklist=["input_dir", "output_dir", "overwrite"])
def reason(
    input_dir,
    output_dir,
    overwrite=False,
    model=gin.REQUIRED,
    num_iterations=gin.REQUIRED,
    training_steps_per_iteration=gin.REQUIRED,
    eval_steps_per_iteration=gin.REQUIRED,
    random_seed=gin.REQUIRED,
    batch_size=gin.REQUIRED,
    name="",
):
  """Trains the estimator and exports the snapshot and the gin config.

  The use of this function requires the gin binding 'dataset.name' to be
  specified if a model is trained from scratch as that determines the data set
  used for training.

  Args:
    byol_config: String with path to BYOL config file. 
    output_dir: String with the path where the results should be saved.
    overwrite: Boolean indicating whether to overwrite output directory.
    model: GaussianEncoderModel that should be trained and exported.
    num_iterations: Integer with number of training steps.
    training_steps_per_iteration: Integer with number of training steps per
      iteration.
    eval_steps_per_iteration: Integer with number of validationand test steps
      per iteration.
    random_seed: Integer with random seed used for training.
    batch_size: Integer with the batch size.
    name: Optional string with name of the model (can be used to name models).
  """
  # We do not use the variable 'name'. Instead, it can be used to name results
  # as it will be part of the saved gin config.
  del name

  # Delete the output directory if it already exists.

  if tf.gfile.IsDirectory(output_dir):
    if overwrite:
      tf.gfile.DeleteRecursively(output_dir)
    else:
      raise ValueError("Directory already exists and overwrite is False.")

  # Create a numpy random state. We will sample the random seeds for training
  # and evaluation from this.
  random_state = np.random.RandomState(random_seed)


  if gin.query_parameter("dataset.name") in ["shapes3d","3dshapes"]:
      epoch = 15
      with gin.unlock_config():
          gin.bind_parameter("dataset.name","shapes3d")
  dataset = pgm_data.get_pgm_dataset()
  # Computes the mean and std of the dataset
  if gin.query_parameter("TwoStageModel.embedding_model_class") not in ['values', 'onehot']:
    std, mean = _compute_std_and_mean()
    with gin.unlock_config():
        gin.bind_parameter("global_stat.mean", mean)
        gin.bind_parameter("global_stat.std", std)


  # We create a TPUEstimator based on the provided model. This is primarily so
  # that we could switch to TPU training in the future. For now, we train
  # locally on GPUs.
  run_config = contrib_tpu.RunConfig(
      tf_random_seed=random_seed,
      keep_checkpoint_max=1,
      tpu_config=contrib_tpu.TPUConfig(iterations_per_loop=500))
  tpu_estimator = contrib_tpu.TPUEstimator(
      use_tpu=False,
      model_fn=model.model_fn,
      model_dir=os.path.join(output_dir, "tf_checkpoint"),
      train_batch_size=batch_size,
      eval_batch_size=batch_size,
      config=run_config)

  # Set up time to keep track of elapsed time in results.
  experiment_timer = time.time()

  # Create a dictionary to keep track of all relevant information.
  results_dict_of_dicts = {}
  validation_scores = []
  all_dicts = []

  for i in range(num_iterations):
    steps_so_far = i * training_steps_per_iteration
    tf.logging.info("Training to %d steps.", steps_so_far)
    # Train the model for the specified steps.
    tpu_estimator.train(
        input_fn=dataset.make_input_fn(random_state.randint(2**32)),
        steps=training_steps_per_iteration)
    # Compute validation scores used for model selection.
    validation_results = tpu_estimator.evaluate(
        input_fn=dataset.make_input_fn(
            random_state.randint(2**32), num_batches=eval_steps_per_iteration))
    validation_scores.append(validation_results["accuracy"])
    tf.logging.info("Validation results %s", validation_results)
    # Compute test scores for final results.
    test_results = tpu_estimator.evaluate(
        input_fn=dataset.make_input_fn(
            random_state.randint(2**32), num_batches=eval_steps_per_iteration),
        name="test")
    dict_at_iteration = results.namespaced_dict(
        val=validation_results, test=test_results)
    results_dict_of_dicts["step{}".format(steps_so_far)] = dict_at_iteration
    all_dicts.append(dict_at_iteration)

  # Select the best number of steps based on the validation scores and add it as
  # as a special key to the dictionary.
  best_index = np.argmax(validation_scores)
  results_dict_of_dicts["best"] = all_dicts[best_index]

  # Save the results. The result dir will contain all the results and config
  # files that we copied along, as we progress in the pipeline. The idea is that
  # these files will be available for analysis at the end.
  original_results_dir = None
  results_dict = results.namespaced_dict(**results_dict_of_dicts)
  results_dir = os.path.join(output_dir, "results")
  results_dict["elapsed_time"] = time.time() - experiment_timer
  results.update_result_directory(results_dir, "abstract_reasoning",
                                  results_dict, original_results_dir)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="abstarct-reasoning-evaluate")
  parser.add_argument('--byol_config', default='config/default.yaml', required = False, help="config file in .yaml format")
  parser.add_argument('--output_dir', default=None, help="Directory to save results into.")
  parser.add_argument('--gin_config', default=[], nargs='+', help="List of paths to the config files.")
  parser.add_argument('--gin_bindings', default=[], nargs='+', help="Newline separated list of Gin parameter bindings.")
  parser.add_argument('--overwrite', action='store_true', default=False, help="Whether to overwrite output directory.")
  parser.add_argument('--reason_index', default=0, choices=list(range(288)), type=int, help="Index in the product sample space to generate reasoning model.")
  parser.add_argument('--edge_mlp_head', default=None, type=int, help="Number of target dim to reduce embeddings to.")

  command_args, unk_args = parser.parse_known_args()
  head_units = command_args.edge_mlp_head

  model_binding_keys = ["AdamOptimizer.learning_rate",
 "OptimizedWildRelNet.dropout_in_last_graph_layer",
 "OptimizedWildRelNet.graph_mlp",
 "OptimizedWildRelNet.edge_mlp",
          ]
  model_binding_candidates = [
        ["1e-2", "1e-3", "1e-4"],# leraning rate
        ["0.25", "0.5", "0.75", "None"],# dropout in last graph layer
        ["[256]", "[128]", "[128, 128]", "[256, 256]"],# graph mlp
        ["[512, 512, 512, 512]", "[512, 512, 512]", "[512, 512]", "[256, 256, 256, 256]", "[256, 256, 256]", "[256, 256]"] if not head_units else \
         [f"[{head_units}, 512, 512, 512, 512]", f"[{head_units}, 512, 512, 512]", f"[{head_units}, 512, 512]", f"[{head_units}, 256, 256, 256, 256]", f"[{head_units}, 256, 256, 256]", f"[{head_units}, 256, 256]"],# edge mlp
        ]
  # The total sample space size is 3*4*4*6=288
  model_binding_values = next(itertools.islice(itertools.product(*model_binding_candidates), command_args.reason_index, command_args.reason_index+1))

  for k,v in zip(model_binding_keys, model_binding_values):
      command_args.gin_bindings.append(f"{k}={v}")

  reason_with_gin(command_args.byol_config, command_args.output_dir, command_args.overwrite,
                         command_args.gin_config, command_args.gin_bindings)
