"""Implementation of Disentanglement, Completeness and Informativeness.
Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
"""
# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, argparse, yaml
from joblib import Parallel, delayed
from pathlib import Path
import torch
import json
import logging
import tensorflow.compat.v1 as tf
import time
import gin.tf
import numpy as np
import pandas as pd
import eval_utils
import random 
from functools import partial
import multiprocessing
import scipy
from six.moves import range
import sklearn
from sklearn import ensemble, linear_model, preprocessing
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
from tensorflow.compat.v1 import gfile
import contextlib

AFF_REGISTRY = {} 
IMPORTANCE_FUNC = lambda x : None
ROTATE_SEED=0

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def aff_register(name):
    def inner_f(fn):
        return AFF_REGISTRY.setdefault(name, fn)
    return inner_f

def get_rotation_mat_QR(z_dim):
    """Generate an orthonomal matrix by QR factorization."""
    with temp_seed(ROTATE_SEED):
        m = np.random.randn(z_dim, z_dim)
        q, _ = np.linalg.qr(m)
    return q

def pick_by_dis_per_factor(importance_matrix, per_dim):
    """For each factor, select `per_dim` most disentangle dim by disentanglement score."""
    latent_num, factor_num= importance_matrix.shape
    dis_per_code = disentanglement_per_code(importance_matrix)
    sort_index = np.argsort(-1 * dis_per_code)
    factor_per_code = np.argmax(importance_matrix, axis=1)
    factor_dim = [[] for _ in range(factor_num)]
    is_full = [False for _ in range(factor_num)]
    for dim in sort_index:
        cur_factor = factor_per_code[dim]
        if len(factor_dim[cur_factor]) < per_dim:
            factor_dim[cur_factor].append(dim)
        else:
            is_full[cur_factor] = True
        if all(is_full) == True:
            break
    select_index = []
    for fac_d in factor_dim:
        select_index.extend(fac_d)
    return list(set(select_index))

@gin.configurable(
    "custom_dci_metric",
    blacklist=["ground_truth_data", "representation_function", "random_state"])
def compute_dci(ground_truth_data, representation_function, random_state,
                artifact_dir=None,
                num_train=gin.REQUIRED,
                num_test=gin.REQUIRED,
                batch_size=gin.REQUIRED,
                aff_type=gin.REQUIRED,
                rotate=gin.REQUIRED,
                save_dir=gin.REQUIRED):
  """Computes the DCI scores according to Sec 2.
  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    num_test: Number of points used for testing.
    batch_size: Batch size for sampling.
    dataset_name: Evaluate dataset name.
  Returns:
    Dictionary with average disentanglement score, completeness and
      informativeness (train and test).
  """
  del artifact_dir
  print("Generating training set.")
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
  if rotate:
    mus_train = np.tile(mus_train, (100,1))
    mus_test = np.tile(mus_test, (100,1))
    q = get_rotation_mat_QR(mus_train.shape[0])
    mus_train = q @ mus_train
    mus_test = q @ mus_test
    scores_full, important_matrix_full, _ = _compute_dci(mus_train, ys_train, mus_test, ys_test)
    dim_ref = pick_by_dis_per_factor(important_matrix_full.T, 2)
    mus_train = mus_train[dim_ref, :]
    mus_test = mus_test[dim_ref, :]
  scores, aff_mat, corr = _compute_dci(mus_train, ys_train, mus_test, ys_test)
  if rotate:
      for k,v in scores_full.items():
          scores[k+'_rotate_orig'] = v
  co_mat, co_score = eval_utils.compute_CO(aff_mat)
  scores["co_score"] = co_score

  dataset_name = gin.query_parameter("dataset.name")
  dataset_info = eval_utils.get_dataset_info(dataset_name)
  with open(os.path.join(save_dir, f'{dataset_name}_{aff_type}_dci_mat.npy'), 'wb') as f:
      np.save(f, aff_mat)
  eval_utils.vis_aff_mat(aff_mat,
          dataset_info=dataset_info,
          save_name=os.path.join(save_dir, f'{dataset_name}_{aff_type}_rotate{rotate}_dci_mat'),
          aff_type='Importance matrix')
  eval_utils.vis_CO(co_mat,
          dataset_info=dataset_info,
          save_name=os.path.join(save_dir, f'{dataset_name}_{aff_type}_rotate{rotate}_dci_CO'),
          aff_type='Importance matrix')
  eval_utils.vis_corr(corr,
          dataset_info=dataset_info,
          save_name=os.path.join(save_dir, f'{dataset_name}_{aff_type}_rotate{rotate}_dci_corr'))
  return scores


def _compute_dci(mus_train, ys_train, mus_test, ys_test):
  """Computes score based on both training and testing codes and factors."""
  scores = {}
  importance_matrix, train_err, test_err, test_err_lst = IMPORTANCE_FUNC(
      mus_train, ys_train, mus_test, ys_test)
  assert importance_matrix.shape[0] == mus_train.shape[0]
  assert importance_matrix.shape[1] == ys_train.shape[0]
  scores["informativeness_train"] = train_err
  scores["informativeness_test"] = test_err
  scores["disentanglement"] = disentanglement(importance_matrix)
  scores["completeness"] = completeness(importance_matrix)
  for fac_idx, test_err in enumerate(test_err_lst):
      scores[f"informativeness_test_factors{fac_idx}"] = test_err
  sorted_mus = mus_test[eval_utils.get_index(importance_matrix.transpose())]
  corr = np.abs(np.corrcoef(sorted_mus))
  return scores, importance_matrix.transpose(), corr

def _gbt_per_factor(X, y):
  """Fit one gbt"""
  model = ensemble.GradientBoostingClassifier() 
  return model.fit(X, y)

def _compute_importance_tree(x_train, y_train, x_test, y_test, model_type = 'gradient_boost'):
  """Compute importance based on gradient boosted trees."""
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  importance_matrix = np.zeros(shape=[num_codes, num_factors],
                               dtype=np.float64)
  train_loss = []
  test_loss = []
  tik = time.time()
  model_list = []
  if model_type is 'gradient_boost':
      model_list = Parallel(n_jobs=num_factors)(delayed(_gbt_per_factor)(x_train.T, y) for y in y_train)
  else:
      for y in y_train:
          model = ensemble.RandomForestClassifier(n_estimators=10, n_jobs=-1)
          model_list.append(model.fit(x_train.T, y))


  for i in range(num_factors):
    #model = ensemble.GradientBoostingClassifier() if model_type is 'gradient_boost'  else ensemble.RandomForestClassifier(n_estimators=10, n_jobs=-1)
    model = model_list[i]
    #In dci paper, the number of trees is 10, max depth is selected depending on validation performance.
    #model = ensemble.RandomForestClassifier(n_estimators=10, n_jobs=-1)
    #model.fit(x_train.T, y_train[i, :])
    importance_matrix[:, i] = np.abs(model.feature_importances_)
    #NOTE: Copy and paste from disentangle_lib. It's acctually train acc here
    train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
    test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))

  tok = time.time()
  logging.info(f'Total running time {tok-tik}s')
  return importance_matrix, np.mean(train_loss), np.mean(test_loss), test_loss

@aff_register('random_forest')
def compute_importance_random_forest(*args, **kwargs):
    return partial(_compute_importance_tree, model_type='random_forest')(*args, **kwargs)

@aff_register('gradient_boost')
def compute_importance_gradient_boost(*args, **kwargs):
    return partial(_compute_importance_tree, model_type='gradient_boost')(*args, **kwargs)

def discrete_mutual_info(mus, ys):
  """Compute discrete mutual information."""
  num_codes = mus.shape[0]
  num_factors = ys.shape[0]
  m = np.zeros([num_codes, num_factors])
  for i in range(num_codes):
    for j in range(num_factors):
      m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
  return m

def histogram_discretize(target, num_bins):
  """Discretization based on histograms."""
  discretized = np.zeros_like(target)
  for i in range(target.shape[0]):
    discretized[i, :] = np.digitize(target[i, :], np.histogram(
        target[i, :], num_bins)[1][:-1])
  return discretized

def discrete_entropy(ys):
  """Compute discrete mutual information."""
  num_factors = ys.shape[0]
  h = np.zeros(num_factors)
  for j in range(num_factors):
    h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
  return h

def _compute_importance_mi(x_train, y_train, x_test, y_test, norm='factor_sum'):
  """Compute importance matrix based on Mutual Information and informativeness based on Logistic."""
  num_factors = y_train.shape[0]
  num_codes = x_train.shape[0]
  q = get_rotation_mat_QR(num_codes)
  x_train = x_train
  discretized_mus = histogram_discretize(x_train, 20)
  m = discrete_mutual_info(discretized_mus, y_train)
  entropy = discrete_entropy(y_train)
  # m's shape is num_codes x num_factors
  if norm == 'factor_sum':
      importance_matrix = np.divide(m, m.sum(axis=0))
  elif norm == 'entropy':
      importance_matrix = np.divide(m, entropy)
  else:
      importance_matrix = np.divide(m, m.sum(axis=0))
      norm = 'factor_sum'


  train_loss = []
  test_loss = []
  tik = time.time()
  for i in range(num_factors):
    model = linear_model.LogisticRegression()
    # Some case fails to converge, add preprocessing to scale data to zero mean
    # and unit std.
    scaler = preprocessing.StandardScaler().fit(x_train.T)
    x_train_scale = scaler.transform(x_train.T)
    x_test_scale = scaler.transform(x_test.T)
    model.fit(x_train_scale, y_train[i, :])
    #NOTE: Copy and paste from disentangle_lib. It's acctually train acc here
    train_loss.append(np.mean(model.predict(x_train_scale) == y_train[i, :]))
    test_loss.append(np.mean(model.predict(x_test_scale) == y_test[i, :]))

  tok = time.time()
  logging.info(f'Total running time {tok-tik}s')
  return importance_matrix, np.mean(train_loss), np.mean(test_loss), test_loss

@aff_register('MI_factor_sum')
def compute_importance_mi_factor_sum(*args, **kwargs):
    return partial(_compute_importance_mi, norm='factor_sum')(*args, **kwargs)

@aff_register('MI_entropy')
def compute_importance_mi_entropy(*args, **kwargs):
    return partial(_compute_importance_mi, norm='entropy')(*args, **kwargs)


def disentanglement_per_code(importance_matrix):
  """Compute disentanglement score of each code."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                  base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
  """Compute the disentanglement score of the representation."""
  per_code = disentanglement_per_code(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

  return np.sum(per_code*code_importance)


def completeness_per_factor(importance_matrix):
  """Compute completeness of each factor."""
  # importance_matrix is of shape [num_codes, num_factors].
  return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                  base=importance_matrix.shape[0])


def completeness(importance_matrix):
  """"Compute completeness of the representation."""
  per_factor = completeness_per_factor(importance_matrix)
  if importance_matrix.sum() == 0.:
    importance_matrix = np.ones_like(importance_matrix)
  factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
  return np.sum(per_factor*factor_importance)

def _load(path):
    with open(path, 'r') as f:
        result = json.load(f) 
    result['path'] = path
    return result

if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  parser = argparse.ArgumentParser(description="dci-metrics-score")
  parser.add_argument('--aff_type', type=str, default='MI', help="metod to compute importance matrix")
  parser.add_argument('--ckpt_path', type=str, default=None, nargs='+', help="TF hub checkpoints path to evaluate")
  parser.add_argument('--rotate', type=int, default=0, help="if set to 1, repeat the representations and rotate")

  command_args = parser.parse_args()
  aff_type = command_args.aff_type
  rotate = command_args.rotate
  IMPORTANCE_FUNC = AFF_REGISTRY.get(aff_type, compute_importance_mi_factor_sum)


  num_train = 10000
  num_test = 10000
  batch_size = 64
  overwrite = True
  base_path = os.path.dirname(command_args.ckpt_path[0].strip('/'))

  for path in command_args.ckpt_path:
      ROTATE_SEED = int(os.path.basename(path.strip(os.sep)))
      gin_bindings = [
              "evaluation.evaluation_fn = @custom_dci_metric",
              f'custom_dci_metric.num_train = {num_train}',
              f'custom_dci_metric.num_test = {num_test}',
              f'custom_dci_metric.save_dir = "{path}"',
              f'custom_dci_metric.batch_size = {batch_size}',
              f'custom_dci_metric.aff_type = "{aff_type}"',
              f'custom_dci_metric.rotate = "{rotate}"',
              "evaluation.random_seed = 0",
              "dataset.name='auto'",
              ]
      result_path = os.path.join(path, "metrics", "custom_dci_metric_" + aff_type + f"_rotate{rotate}")
      representation_path = os.path.join(path, "representation")
      model_path = os.path.join(path, "model")
      postprocess_gin = ["postprocess.gin"]
      postprocess.postprocess_with_gin(model_path, representation_path, overwrite, postprocess_gin)
      evaluate.evaluate_with_gin(
             representation_path, result_path, overwrite, gin_bindings=gin_bindings)

  pattern = os.path.join(base_path,
          f"*/metrics/custom_dci_metric_{aff_type}_rotate{rotate}/results/json/evaluation_results.json")
  results_path = os.path.join(command_args.ckpt_path[0], aff_type+f'_rotate{rotate}_force_aggregate_results.json')

  res_files = Path('.').glob(pattern)
  pool = multiprocessing.Pool()
  all_results = pool.map(_load, res_files)
  res_df = pd.DataFrame(all_results)
  final_res = pd.DataFrame({'mean':res_df.mean(), 'std':res_df.std()})
  with open(results_path, 'w') as f:
      final_res.to_json(path_or_buf=f, orient='index')


