import pdb

import numpy as np
import scipy.io
import scipy.ndimage
import sklearn.preprocessing
import pandas as pd

class Dataset():
    """ 
    Dataset manager class
    """
    def  __init__(self, data_path=None):
        """
        Class intitializer.
        """
        # set database path
        if data_path == None:
            self.data_path = './run_data/'
        # feature and target vecotrs
        self.features = None
        self.targets = None
        self.order = None
        self.mask = None
        self.costs = None


    def load(self, dataset_name, options=None):
        """
        Load dataset by name.
        """
        if dataset_name == 'mnist':
            self.load_mnist(options)
        elif dataset_name == 'synthesized':
            self.load_synthesized(options)
        else:
            raise NotImplementedError('dataset loader not found.')


    def get(self, order='none', onehot=True):
        """
        Get dataset fields.
        """
        # create the reqested ordering
        if order == 'rand':
            self.order = np.random.permutation(self.features.shape[0])
            self.features = self.features[self.order]
            self.targets = self.targets[self.order]

        # create the onehot representation
        if onehot:
            enc = sklearn.preprocessing.OneHotEncoder(sparse=False)
            new_targets = enc.fit_transform(self.targets)
        else:
            new_targets = self.targets

        return {'features':self.features, 
                'targets':new_targets, 
                'order':self.order,
                'mask':self.mask,
                'costs':self.costs}
    

    def preprocess(self, normalization='none', fe_std_threshold=0.0):
        # apply feature std threshold
        fe_mask = self.features.std(axis=0) >= fe_std_threshold
        self.features = self.features[:,fe_mask]
        self.mask = fe_mask
        self.costs = self.costs[fe_mask]

        # apply different normalizations
        if normalization == 'none':
            pass
        elif normalization == 'center':
            self.features = self.features - self.features.mean(axis=0)
        elif normalization == 'unity':
            self.features = (self.features - self.features.min(axis=0))\
                    / (self.features.max(axis=0) - self.features.min(axis=0))
        elif normalization == 'stat':
            self.features = (self.features - self.features.mean(axis=0))\
                    / self.features.std(axis=0)

    
    def load_synthesized(self, opts):
        N_FEATURES = opts['n_features']
        N_CLUSTERS = opts['n_clusters']
        N_CLUSTERPOINTS = opts['n_clusterpoints']
        STD_CLUSTERS = opts['std_clusters']
        np.random.seed(1)
        dataset_features = []
        dataset_targets = []
        cluster_labels = np.random.permutation([1]*(N_CLUSTERS//2) + [2]*(N_CLUSTERS//2))
            
        for clus in range(N_CLUSTERS):
            pos_center = np.random.rand(N_FEATURES)
            label_cluser = cluster_labels[clus]
            dataset_features.append(pos_center + STD_CLUSTERS*np.random.randn(N_CLUSTERPOINTS,N_FEATURES))
            dataset_targets.append(np.ones((N_CLUSTERPOINTS,1), dtype=np.float) * label_cluser)

        dataset_features = np.vstack(dataset_features)
        dataset_targets = np.vstack(dataset_targets)

        # random permutation
        inds_sel = np.random.permutation(dataset_features.shape[0])
        dataset_features = dataset_features[inds_sel,:]
        dataset_targets = dataset_targets[inds_sel,:]

        # set attributes
        if 'cost-aware' in opts and opts['cost-aware'] == True:
            redundant_features = np.random.randn(dataset_features.shape[0], dataset_features.shape[1])
            self.features = np.hstack([dataset_features, redundant_features])
            self.targets = dataset_targets.reshape(-1,1)
            self.order = np.arange(self.features.shape[0])
            self.costs = np.hstack([np.arange(1, dataset_features.shape[1]+1, dtype=np.float), 
                                   np.arange(1, redundant_features.shape[1]+1, dtype=np.float)])
        else:
            self.features = dataset_features
            self.targets = dataset_targets.reshape(-1,1)
            self.order = np.arange(self.features.shape[0])
            self.costs = np.ones((self.features.shape[1],), dtype=np.float)
    
    def load_mnist(self, opts):
        # load and read data
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("./run_data/MNIST_data/", one_hot=False)
        if opts == None or opts['task'] == 'singleres':
            self.features = mnist.train.images
            self.targets = mnist.train.labels.reshape(-1,1)
            self.order = np.arange(self.features.shape[0])
            self.costs = np.ones((self.features.shape[1],), dtype=np.float)
            return
        # check if it is a multires cost-sensitive case
        elif opts['task'] == 'multires':
            images = mnist.train.images
            self.targets = mnist.train.labels.reshape(-1,1)
            self.features = np.hstack([images, 
                                       resize_images(images, 0.5), 
                                       resize_images(images, 0.25), 
                                       resize_images(images, 0.125)])
            self.targets = mnist.train.labels.reshape(-1,1)
            self.order = np.arange(self.features.shape[0])
            self.costs = np.hstack([np.ones((images.shape[1],), dtype=np.float) * 4.0, 
                                   np.ones((resize_images(images[:2,:], 0.500).shape[1],), dtype=np.float) * 3.0, 
                                   np.ones((resize_images(images[:2,:], 0.250).shape[1],), dtype=np.float) * 2.0,
                                   np.ones((resize_images(images[:2,:], 0.125).shape[1],), dtype=np.float) * 1.0,])
            return
        else:
            raise NotImplementedError('task not found!')

    
def resize_images(images, scale):
    dimxy = int(np.sqrt(images.shape[1]))
    resized_images = []
    for img in images:
        img_new = scipy.ndimage.interpolation.zoom(img.reshape(dimxy,dimxy), scale)
        resized_images.append(img_new.reshape(1, -1))
    resized_images = np.vstack(resized_images)
    return resized_images

