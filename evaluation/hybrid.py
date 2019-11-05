import os
import sys

import numpy as np 

eps_ = 1e-15

def normalize(x, axis):
    return x / np.linalg.norm(x, axis=axis, keepdims=True)

def cosine_similarity(pred, prototypes):
    nsamples, ndims = np.shape(pred)
    nclasses, _ = np.shape(prototypes)

    pred = pred[:, np.newaxis, :].repeat(nclasses, axis=1)
    prototypes = prototypes[np.newaxis, :, :]

    res = np.sum(normalize(pred, 2) * normalize(prototypes, 2), 2) 
    return res

def eucledian_distance(pred, prototypes):
    nsamples, ndims = np.shape(pred)
    nclasses, _ = np.shape(prototypes)

    pred = pred[:, np.newaxis, :].repeat(nclasses, axis=1)
    prototypes = prototypes[np.newaxis, :, :]
    res = np.sum((pred - prototypes) ** 2, 2)
    return res

def hybrid_labeling(pred_attr, pred_latent, attr_prototype, latent_prototype, ensemble=True, metric='cosine'):
    cos_sim_attr = cosine_similarity(pred_attr, attr_prototype)

    if metric == "cosine":
        latent_sim = cosine_similarity(pred_latent, latent_prototype)
    elif metric == 'euclidean':
        latent_sim = 1 / (1 + eucledian_distance(pred_latent, latent_prototype))
        cos_sim_attr = normalize(cos_sim_attr, 1)
        latent_sim = normalize(latent_sim, 1)
    
    if ensemble:
        final_sim = cos_sim_attr + latent_sim 
    else:
        final_sim = latent_sim
    return final_sim.argmax(1)
