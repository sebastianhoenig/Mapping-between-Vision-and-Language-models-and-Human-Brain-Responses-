""" Utility functions for analysis """

import os
import torch
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import numpy as np
import re
from tqdm import tqdm


def mean_vilt(vilt_layers, layer_name=""):
    mean_layers = {}
    for i, layer in enumerate(vilt_layers):
        layer = torch.split(layer, split_size_or_sections=10, dim=0)
        mean_layer = [torch.mean(t, dim=0, keepdim=True) for t in layer]
        mean_layer = torch.stack(mean_layer, dim=0).squeeze()
        mean_layer = mean_layer.detach().cpu().numpy()
        if layer_name:
            k = f"VILT {layer_name} Layer {i+1}"
        else:
            k = f"VILT Layer {i+1}"
        mean_layers[k] = mean_layer
    return mean_layers


def mean_vit(vit_layers):
    mean_layers = {}
    for i, layer in enumerate(vit_layers):
        layer = torch.split(layer, split_size_or_sections=10, dim=0)
        mean_layer = [torch.mean(t, dim=0, keepdim=True) for t in layer]
        mean_layer = torch.stack(mean_layer, dim=0).squeeze()
        mean_layer = mean_layer.detach().cpu().numpy()
        k = f"Vit Layer {i + 1}"
        mean_layers[k] = mean_layer
    return mean_layers


def mean_clip(clip_layers, split_size, layer_name=""):
    mean_layers = {}
    for i, layer in enumerate(clip_layers):
        layer = torch.split(layer, split_size_or_sections=split_size, dim=0)
        mean_layer = [torch.mean(t, dim=0, keepdim=True) for t in layer]
        mean_layer = torch.stack(mean_layer, dim=0).squeeze()
        mean_layer = mean_layer.detach().cpu().numpy()
        k = f"CLIP {layer_name} Layer {i+1}"
        mean_layers[k] = mean_layer
    return mean_layers


def mean_bert(bert_layers, split_size, layer_name=""):
    mean_layers = {}
    for i, layer in enumerate(bert_layers):
        layer = torch.split(layer, split_size_or_sections=split_size, dim=0)
        mean_layer = [torch.mean(t, dim=0, keepdim=True) for t in layer]
        mean_layer = torch.stack(mean_layer, dim=0).squeeze()
        mean_layer = mean_layer.detach().cpu().numpy()
        k = f"Bert {layer_name} Layer {i+1}"
        mean_layers[k] = mean_layer
    return mean_layers


def mean_gpt(bert_layers, split_size, layer_name=""):
    mean_layers = {}
    for i, layer in enumerate(bert_layers):
        layer = torch.split(layer, split_size_or_sections=split_size, dim=0)
        mean_layer = [torch.mean(t, dim=0, keepdim=True) for t in layer]
        mean_layer = torch.stack(mean_layer, dim=0).squeeze()
        mean_layer = mean_layer.detach().cpu().numpy()
        k = f"GPT {layer_name} Layer {i+1}"
        mean_layers[k] = mean_layer
    return mean_layers


def mean_albef(albef_layers, layer_name=""):
    mean_layers = {}
    for i, layer in enumerate(albef_layers):
        layer = torch.split(layer, split_size_or_sections=10, dim=0)
        mean_layer = [torch.mean(t, dim=0, keepdim=True) for t in layer]
        mean_layer = torch.stack(mean_layer, dim=0).squeeze()
        mean_layer = mean_layer.detach().cpu().numpy()
        k = f"ALBEF {layer_name} Layer {i+1}"
        mean_layers[k] = mean_layer
    return mean_layers


def load_files(dir_path, regex=None):
    # Get list of all files in the directory
    file_list = os.listdir(dir_path)
    # order list alphabetically
    file_list = sorted(file_list)
    if regex:
        file_list = [f for f in file_list if re.match(regex, f)]
    tensor_list = []
    # Loop over all files
    for file_name in file_list:
        if not file_name.endswith('.pt'):
            continue
        # Load the file and convert it to a tensor
        file_path = os.path.join(dir_path, file_name)
        tensor = torch.load(file_path, map_location=torch.device('cpu'))
        # Add the tensor to the list
        tensor_list.append(tensor)
    # Stack all tensors
    result = torch.stack(tensor_list, dim=0)
    return result


def inter_intra_similarity(feats, layer_names):
    d = {}
    for i, layer in enumerate(feats):
        k = f"{layer_names} {i + 1}"
        d[k] = layer.detach().cpu().numpy()
    rdms = get_rdms(d)

    all_intra_cluster_similarities = []
    all_inter_cluster_similarities = []

    for o in range(1, len(rdms.keys()) + 1):
        layer = rdms[f'{layer_names} {o}']

        diagonal_sum = 0
        intra_cluster_similarities = []
        for i in range(81):
            cluster_submatrix = layer[i * 10:(i + 1) * 10, i * 10:(i + 1) * 10]
            cluster_submatrix[cluster_submatrix < 1e-5] = 0
            cluster_submatrix_no_diagonal = cluster_submatrix[~np.eye(10, dtype=bool)]
            intra_cluster_similarities.append(np.mean(cluster_submatrix_no_diagonal))
            diagonal_sum += np.sum(cluster_submatrix_no_diagonal)

        matrix_sum = np.sum(layer)
        matrix_sum -= diagonal_sum
        matrix_mean = matrix_sum / (810 * 810 - 8100)

        average_intra_cluster_similarity = 1-np.mean(intra_cluster_similarities)
        average_inter_cluster_similarity = 1-matrix_mean

        all_intra_cluster_similarities.append(average_intra_cluster_similarity)
        all_inter_cluster_similarities.append(average_inter_cluster_similarity)

    return all_intra_cluster_similarities, all_inter_cluster_similarities


def prepare_vilt(vilt_layers, layer_name=""):
    prepd_layers = {}
    for i, layer in enumerate(vilt_layers):
        layer = layer.detach().cpu().numpy()
        if layer_name:
            k = f"VILT {layer_name} Layer {i+1}"
        else:
            k = f"VILT Layer {i+1}"
        prepd_layers[k] = layer
    return prepd_layers


def prepare_vit(vit_layers):
    prepd_layers = {}
    for i, layer in enumerate(vit_layers):
        layer = layer.detach().cpu().numpy()
        k = f"Vit Layer {i + 1}"
        prepd_layers[k] = layer
    return prepd_layers


def prepare_clip(clip_layers, layer_name=""):
    prepd_layers = {}
    for i, layer in enumerate(clip_layers):
        k = f"CLIP {layer_name} Layer {i + 1}"
        layer = layer.detach().cpu().numpy()
        prepd_layers[k] = layer
    return prepd_layers


def prepare_bert(bert_layers, layer_name=""):
    prepd_layers = {}
    for i, layer in enumerate(bert_layers):
        k = f"Bert {layer_name} Layer {i+1}"
        layer = layer.detach().cpu().numpy()
        prepd_layers[k] = layer
    return prepd_layers


def prepare_gpt(gpt_layers, layer_name=""):
    prepd_layers = {}
    for i, layer in enumerate(gpt_layers):
        layer = layer.detach().cpu().numpy()
        k = f"GPT {layer_name} Layer {i+1}"
        prepd_layers[k] = layer
    return prepd_layers


def prepare_albef(albef_layers, layer_name=""):
    prepd_layers = {}
    for i, layer in enumerate(albef_layers):
        layer = layer.detach().cpu().numpy()
        k = f"ALBEF {layer_name} Layer {i+1}"
        prepd_layers[k] = layer
    return prepd_layers


def normalize(rdm):
    rdm = (rdm - rdm.min()) / (rdm.max() - rdm.min())
    return rdm


def get_rdm(feat):
    distance = pdist(feat, metric='cosine')
    rdm = squareform(distance)
    normalized_rdm = normalize(rdm)
    return normalized_rdm


def get_rdms(feats_dict):
    for k, v in feats_dict.items():
        feats_dict[k] = get_rdm(v)
    return feats_dict


def get_upper_triu(matrix, size):
    return matrix[np.triu_indices(size, k=1)]


def get_spearmanr(dict1, size):
    result_dict = {}
    result_dict_no_p = {}
    for k, v in tqdm(dict1.items()):
        v = get_upper_triu(v, size)
        result_dict[k] = {}
        result_dict_no_p[k] = {}
        for k2, v2 in dict1.items():
            v2 = get_upper_triu(v2, size)
            res = stats.spearmanr(v, v2)
            result_dict[k][k2] = (res.correlation, res.pvalue)
            result_dict_no_p[k][k2] = res.correlation
    return result_dict, result_dict_no_p


def add_feature_extractor(model, layers):
    def get_activation(layer_name):
        def hook(_, input, output):
            # clip specific
            if ('visual.transformer.resblocks' in layer_name and 'attn' in layer_name) or \
                    ('transformer.resblocks' in layer_name and 'attn' in layer_name):
                output = output[0]
            # vilt specific
            elif len(output) == 1 and isinstance(output[0], torch.Tensor):
                output = output[0]
            # albef specific
            elif len(output) == 2 and isinstance(output[0][0], torch.Tensor):
                output = output[0][0]
            if not hasattr(model, '__layer_activations__'):
                model.__layer_activations__ = {}
            model.__layer_activations__[layer_name] = output.detach()
        return hook
    for layer_name, layer in model.named_modules():
        if layer_name in layers:
            layer.register_forward_hook(get_activation(layer_name))
    return model


def _get_prompt(concept):
    c_art = 'an' if concept.startswith(('a', 'e', 'i', 'o', 'u')) else 'a'
    prompts = [
        f'a photo of {c_art} {concept}',
    ]
    return prompts


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def sort_list_by_last_word(lst):
    lst.sort(key=lambda x: x.split()[1])
    return lst
