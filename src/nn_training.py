import pdb

import numpy as np
import sklearn.preprocessing


def feed_data(ph_input_features, ph_input_mask, 
              ph_output_targets, ph_input_features_full, 
              dataset_features, dataset_targets, 
              phase, experiment='real_mask', 
              size_batch=100, missing_portion=(1,1), seed=None):
    inds_tst = np.arange(1,dataset_features.shape[0]*0.15, dtype=np.int)
    inds_val = np.arange(dataset_features.shape[0]*0.15, 
                          dataset_features.shape[0]*0.30, dtype=np.int)
    inds_trn = np.arange(dataset_features.shape[0]*0.30, 
                          dataset_features.shape[0]*1.00, dtype=np.int)
    # if the phase is validation
    if phase == 'validation':
        phase_features = dataset_features[inds_val,:]
        phase_targets = dataset_targets[inds_val,:]
    # if the phase is  test
    elif phase == 'test':
        phase_features = dataset_features[inds_tst,:]
        phase_targets = dataset_targets[inds_tst,:]
    # if the phase is  train
    elif phase == 'train':
        phase_features = dataset_features[inds_trn,:]
        phase_targets = dataset_targets[inds_trn,:]
    elif phase == 'all':
        phase_features = dataset_features[:,:]
        phase_targets = dataset_targets[:,:]
    else:
        raise NotImplementedError('phase not found.')    
    
    if phase == 'all':
        # select from the begining
        features = phase_features[:size_batch,:]
        targets = phase_targets[:size_batch,:]
    else: 
        # select a random batch
        np.random.seed(seed)
        inds_sel = np.random.permutation(phase_features.shape[0])[:size_batch]
        features = phase_features[inds_sel,:]
        targets = phase_targets[inds_sel,:]

    
    # inject missing values
    # if input is tuple, use latent beta distribution
    if type(missing_portion) == tuple:
        zs = np.random.beta(missing_portion[0], missing_portion[1], 
                            (features.shape[0],1))
    # else use fixed probability random 
    else:
        zs = missing_portion
    
    mask = np.random.rand(features.shape[0],
                          features.shape[1]) >= zs
    features_full = np.copy(features)
    features[np.logical_not(mask)] = 0
    
    # see if it is a dummy mask
    if experiment == 'ones_mask':
        mask = np.ones(features.shape, dtype=np.float)
    
    return {ph_input_features:features, 
           ph_input_mask:mask, 
           ph_output_targets:targets, 
           ph_input_features_full:features_full}

def feed_data_nhanes(ph_input_features, ph_input_mask, 
              ph_output_targets, ph_input_features_full, 
              dataset_features, dataset_targets, 
              phase, experiment='real_mask', 
              size_batch=100, missing_portion=(1,1), seed=None):
    # preliminary checks
    n_classes = dataset_targets.shape[1]
    
    inds_tst = np.arange(1,dataset_features.shape[0]*0.15, dtype=np.int)
    inds_val = np.arange(dataset_features.shape[0]*0.15, 
                          dataset_features.shape[0]*0.30, dtype=np.int)
    inds_trn = np.arange(dataset_features.shape[0]*0.30, 
                          dataset_features.shape[0]*1.00, dtype=np.int)
    # if the phase is validation
    if phase == 'validation':
        phase_features = dataset_features[inds_val,:]
        phase_targets = dataset_targets[inds_val,:]
    # if the phase is  test
    elif phase == 'test':
        phase_features = dataset_features[inds_tst,:]
        phase_targets = dataset_targets[inds_tst,:]
    # if the phase is  train
    elif phase == 'train':
        phase_features = dataset_features[inds_trn,:]
        phase_targets = dataset_targets[inds_trn,:]
    elif phase == 'all':
        phase_features = dataset_features[:,:]
        phase_targets = dataset_targets[:,:]
    else:
        raise NotImplementedError('phase not found.')    
    
    # select a random and balanced batch
    phase_targets_dense = np.argmax(phase_targets, 1)
    np.random.seed(seed)
    inds_perm = np.random.permutation(phase_features.shape[0])
    inds_sel = []
    for cl in range(n_classes):
        inds_cl = inds_perm[phase_targets_dense[inds_perm] == cl]
        inds_sel.extend(inds_cl[:size_batch//n_classes])
    inds_sel = np.random.permutation(inds_sel)
    features = phase_features[inds_sel,:]
    targets = phase_targets[inds_sel,:]
    
    # inject missing values
    # if input is tuple, use latent beta distribution
    if type(missing_portion) == tuple:
        zs = np.random.beta(missing_portion[0], missing_portion[1], 
                            (features.shape[0],1))
    # else use fixed probability random 
    else:
        zs = missing_portion
    
    mask = np.random.rand(features.shape[0],
                          features.shape[1]) >= zs
    features_full = np.copy(features)
    features[np.logical_not(mask)] = 0
    
    # see if it is a dummy mask
    if experiment == 'ones_mask':
        mask = np.ones(features.shape, dtype=np.float)
    
    return {ph_input_features:features, 
           ph_input_mask:mask, 
           ph_output_targets:targets, 
           ph_input_features_full:features_full}


