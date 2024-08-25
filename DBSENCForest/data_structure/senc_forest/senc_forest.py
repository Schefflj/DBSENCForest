import numpy as np
from time import time

from DBSENCForest.data_structure.senc_forest.senc_tree import SENCTree


class SENCForest:

    def __init__(self, data, num_tree, num_sub, num_dim, rseed, label):
        """
        Initializes the SENCForest
        :param data: numpy array, input data.
        :param num_tree: int, number of trees in the forest.
        :param num_sub: int, number of subsamples per tree.
        :param num_dim: int, number of dimensions for splitting.
        :param rseed: int, random seed for reproducibility.
        :param label: numpy array, labels for the data.
        """
        self.num_tree = num_tree
        self.num_sub = num_sub
        self.num_dim = num_dim
        self.c = 2 * (np.log(num_sub - 1) + 0.5772156649) - 2 * (num_sub - 1) / num_sub
        self.rseed = rseed
        self.anomaly = []
        np.random.seed(int(rseed))

        self.trees = []
        self.fruit = np.unique(label)
        self.height_limit = 200

        paras = {
            'HeightLimit': self.height_limit,
            'IndexDim': np.arange(data.shape[1]),
            'NumDim': num_dim
        }

        class_index = {j: np.where(label == fruit)[0] for j, fruit in enumerate(self.fruit)}

        self.build_forest(data, class_index, paras, label)

    def build_forest(self, data, class_index, paras, label):
        """
        Builds the SENCForest.

        :param data: numpy array, input data.
        :param class_index: dict, index of the classes.
        :param paras: dict, parameters for the tree.
        :param label: numpy array, labels for the data.
        :return: None
        """
        start_time = time()
        for i in range(self.num_tree):
            index_sub = []
            for j in range(len(self.fruit)):
                tempin = class_index[j]
                if len(tempin) < self.num_sub:
                    print('Number of instances is too small.')
                else:
                    tempso = np.random.permutation(tempin)
                    index_sub.extend(tempso[:self.num_sub])

            globals_dict = {
                'flag1': 0,
                'id': 1,
                'pathline': [],
                'pathline3': np.empty((0, 1))
            }

            tree = SENCTree(data, index_sub, 0, paras, label, globals_dict)
            tree.totalid = globals_dict['id'] - 1
            tree.pathline = globals_dict['pathline3']
            tree.pathline1 = globals_dict['pathline']

            try:
                tempan = np.sort(np.array(tree.pathline1)[:, 0])
            except:
                print("Warning: tempan does not have enough elements.")
                self.anomaly = None
                continue
            if len(tempan) > 1:
                vars_rate2 = [abs(np.std(tempan[:j]) - np.std(tempan[j:])) for j in range(1, len(tempan))]
                if vars_rate2:
                    bb = np.argmin(vars_rate2)
                    self.anomaly.append(tempan[bb])
                else:
                    print("Warning: vars_rate2 is empty.")
                    self.anomaly = None
            else:
                print("Warning: tempan does not have enough elements.")
                self.anomaly = None

            self.trees.append(tree)

        self.elapse_time = time() - start_time
