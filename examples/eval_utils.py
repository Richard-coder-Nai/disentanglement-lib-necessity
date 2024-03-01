import logging, yaml
import torch
import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os, random 


import tensorflow_hub as hub

def transform_dsprite(input_data):
    input_data = torch.Tensor(input_data)
    input_data = input_data.repeat(1,1,1,3)
    input_data = torch.transpose(input_data, 1, 3)
    return input_data

def transform_3dshapes(input_data):
    input_data = torch.Tensor(input_data)
    input_data = torch.transpose(input_data, 1, 3)
    return input_data

def transform_smallnorb(input_data):
    input_data = torch.Tensor(input_data)
    input_data = input_data.repeat(1,1,1,3)
    input_data = torch.transpose(input_data, 1, 3)
    return input_data 

def get_dataset_info(dataset_name='dsprites_full'):
    name2info = {}
    name2info["3dshapes"] = dict(name='3dshapes', 
            select_index=[0, 1, 2, 3, 4, 5],
            factor_name=['floor color', 'wall color', 'object color', 'object size', 'object type', 'azimuth'],
            transform_fn=transform_3dshapes)
    name2info["cars3d"] = dict(name='cars3d',
            select_index=[0, 1, 2],
            factor_name=['elevation', 'azimuth', 'object type'],
            transform_fn=transform_3dshapes)
    name2info["smallnorb"] = dict(name='smallnorb',
            select_index=[0, 1, 2, 3],
            factor_name=['azimuth', 'object category', 'elevation', 'lighting'],
            transform_fn=transform_smallnorb)
    name2info["dsprites_full"] = dict(name='dsprites_full',
            select_index=[0, 1, 3, 4],
            factor_name=['Shape', 'Scale','Position_X', 'Position_Y'],
            transform_fn=transform_dsprite)
    default_dict = name2info["dsprites_full"]
    return name2info.get(dataset_name, default_dict)
    

def generate_batch_factor_code(ground_truth_data, representation_function,
                               num_points, random_state, batch_size, dataset_info):
    representations = None
    factors = None
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_factors, current_observations = \
                ground_truth_data.sample(num_points_iter, random_state)
        current_observations = dataset_info["transform_fn"](current_observations)
        if i == 0:
            factors = current_factors
            representations = representation_function(current_observations).detach().numpy()
        else:
            factors = np.vstack((factors, current_factors))
            representations = np.vstack((representations, 
                representation_function(
                    current_observations).detach().numpy()))
        i += num_points_iter
    representations = representations.reshape(representations.shape[0], representations.shape[1])
    return representations.transpose(), factors.transpose()


def get_index(m):
    factor_num = m.shape[0]
    latent_num = m.shape[1]

    m_sort_score = np.zeros([latent_num])
    code_importance = m.sum(axis=0) / m.sum()
    thresh = np.percentile(code_importance, 20)
    m_sort_score = np.argmax(m, axis=0) + code_importance
    sort_index = np.argsort(m_sort_score)
    return sort_index


def compute_CO(m):
    """
    Compute co-occurence matrix.
    m: Importance matrix, np.array of shape(factor_num, latent_num)
    Returns: 
        co-occurence matrix
        co-occurence score: 1 - mean(off-diagnal elements)
    """
    factor_num = m.shape[0]
    latent_num = m.shape[1]

    co_mat = np.zeros([factor_num, factor_num])
    for i in range(factor_num):
        for j in range(factor_num):
            den = np.linalg.norm(m[i]) * np.linalg.norm(m[j])
            co_mat[i,j] = np.dot(m[i] , m[j]) / den

    co_score = 1 - np.mean(co_mat[~np.eye(factor_num, dtype=bool)])
    return co_mat, co_score

def vis_aff_mat(m, dataset_info, save_name, aff_type='mutual information'):
    """
    Visualize affinity matrix of factors and representation. 
    m: np.array of shape(factor_num, latent_num)
    dataset_info: dictionary:
        name: str, dataset name
        select_index: list, factor index to visualize
        factor_name:list, factor names 
    """

    # prepare
    
    m = m[dataset_info['select_index']] 
    factor_num = m.shape[0]
    latent_num = m.shape[1]
    # get dataset facotrs name
    
   
    # sort
    sort_index = get_index(m)
    m = m[:, sort_index]

    # plot
    sns.set(rc={"figure.figsize":(50, 12), "font.size":40})
    # TODO install times new roman
    # sns.set_theme(font='Times New Roman')
    htmap = sns.heatmap(m)
    htmap.set_title(f'{aff_type} of factors and latent dimensions'.capitalize(), fontsize = 50)
    htmap.set_xlabel('latent dimension index', fontsize = 40)
    htmap.set_ylabel ('Factor', fontsize = 40)
    plt.xticks(fontsize=30,rotation=0)
    plt.yticks(fontsize=30,rotation=90)
    cbar = htmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)

    # plt.colorbar.ax.tick_params(labelsize=30)
    for ind, label in enumerate(htmap.get_xticklabels()):
        if ind % 10 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)

    legend_str = '\n'.join([f'{factor_idx} - {factor_name}' for factor_idx, factor_name in enumerate(dataset_info["factor_name"])])
    plt.gcf().text(0.02, 0.5, legend_str, fontsize=40)
    plt.savefig(save_name+'.pdf', bbox_inches='tight')
    plt.savefig(save_name+'.png', bbox_inches='tight')
    plt.close()
    return

def vis_CO(m, dataset_info, save_name, aff_type='mutual information'):
    """
    Visualize Co-occurence matrix.
    m: Importance matrix, np.array of shape(factor_num, latent_num)
    dataset_info: dictionary:
        name: str, dataset name
        select_index: list, factor index to visualize
        factor_name:list, factor names 
    """
    # plot
    m = m[dataset_info['select_index']]
    co_mat, _ = compute_CO(m)
    sns.set(rc={"figure.figsize":(6*1.2, 5*1.2)})
    # sns.set_theme(font='Times New Roman')
    htmap = sns.heatmap(co_mat)
    htmap.set_title(f'Co-occurrence of {aff_type}', fontsize = 25)
    htmap.set_ylabel('Factor', fontsize = 20)
    htmap.set_xlabel ('Factor', fontsize = 20)
    plt.xticks(fontsize=15,rotation=0)
    plt.yticks(fontsize=15,rotation=90)
    cbar = htmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    legend_str = '\n'.join([f'{factor_idx} - {factor_name}' for factor_idx, factor_name in enumerate(dataset_info["factor_name"])])
    plt.gcf().text(0.02, 0.5, legend_str, fontsize=14)
    plt.subplots_adjust(left=0.3)
    plt.savefig(save_name+'.pdf', bbox_inches='tight')
    plt.savefig(save_name+'.png', bbox_inches='tight')
    plt.close()
    return

def vis_corr(corr, dataset_info, save_name, eval_metric = 'mig'):
    """
    Visualize mutual neuro-wise corr matrix
    m: np.array of shape(factor_num, latent_num)
    dataset_info: dictionary:
        name: str, dataset name
        select_index: list, factor index to visualize
        factor_name:list, factor names 
    """

    # prepare
    latent_num = corr.shape[0]
    

    # plot
    sns.set(rc={"figure.figsize":(50, 50), "font.size":20})
    # TODO install times new roman
    # sns.set_theme(font='Times New Roman')
    htmap = sns.heatmap(corr)
    htmap.set_title('Correlation of representations', fontsize = 50)
    htmap.set_xlabel('latent dimension index', fontsize = 20)
    htmap.set_ylabel('latent dimension index', fontsize = 20)
    plt.xticks(fontsize=30,rotation=0)
    plt.yticks(fontsize=30,rotation=90)
    cbar = htmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)

    # plt.colorbar.ax.tick_params(labelsize=30)
    for ind, label in enumerate(htmap.get_xticklabels()):
        if ind % 10 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)

    for ind, label in enumerate(htmap.get_yticklabels()):
        if ind % 10 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.savefig(save_name+'.pdf', bbox_inches='tight')
    plt.savefig(save_name+'.png', bbox_inches='tight')
    plt.close()
    return
