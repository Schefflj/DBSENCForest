import numpy as np
from scipy.spatial.distance import cdist

class SENCTree:

    def split_indices(self, curt_index, curt_data, split_point):
        """
        Splits the current indices into left and right based on the split point.

        :param curt_index: numpy array, indices of the current data points.
        :param curt_data: numpy array, values of the current data points for the selected attribute.
        :param split_point: float, the point at which to split the data.
        :return: tuple of numpy arrays, indices for the left and right splits.
        """
        left_curt_index = []
        right_curt_index = []
        for idx, value in zip(curt_index, curt_data):
            if value < split_point:
                left_curt_index.append(idx)
            else:
                right_curt_index.append(idx)
        return np.array(left_curt_index), np.array(right_curt_index)

    def __init__(self, data, curt_index, curt_height, paras, all_train_data_label, globals_dict):
        """
        Initializes a SENCTree node.

        :param data: numpy array, input data used for tree construction.
        :param curt_index: numpy array, indices of the current data points.
        :param curt_height: int, current height of the node in the tree.
        :param paras: dict, parameters for tree construction (e.g., height limit, index dimensions).
        :param all_train_data_label: numpy array, labels for the training data.
        :param globals_dict: dict, global variables shared across the tree.
        """
        self.height = curt_height
        self.size = len(curt_index)

        if curt_height >= paras['HeightLimit'] or self.size <= 10:
            if self.size > 1:
                self.node_status = 0
                self.split_attribute = None
                self.split_point = None
                self.left_child = None
                self.right_child = None
                self.curt_index = curt_index
                self.la = all_train_data_label[curt_index]

                self.id = globals_dict['id']
                globals_dict['id'] += 1
                self.center = np.mean(data[curt_index], axis=0)
                self.dist = np.max(cdist(data[curt_index], [self.center]))

                if self.size != 1:
                    c = 2 * (np.log(self.size - 1) + 0.5772156649) - 2 * (self.size - 1) / self.size
                else:
                    c = 0

                self.high = curt_height + c
                globals_dict['pathline'].append((self.high, self.size))
                globals_dict['pathline3'] = np.concatenate([globals_dict['pathline3'], np.array([self.high] * self.size).reshape(-1, 1)], axis=0)

                if self.size == 1:
                    globals_dict['flag1'] = 1

            else:
                self.node_status = 0
                self.split_attribute = None
                self.split_point = None
                self.left_child = None
                self.right_child = None
                self.curt_index = curt_index
                self.la = []
                if self.size == 1:
                    self.la = all_train_data_label[curt_index]
                    self.center = np.mean(data[curt_index], axis=0)
                self.id = globals_dict['id']
                globals_dict['id'] += 1
                globals_dict['flag1'] = 1
                self.high = curt_height

        else:
            self.node_status = 1
            self.split_attribute = np.random.choice(paras['IndexDim'])
            curt_data = data[curt_index, self.split_attribute]
            self.split_point = np.min(curt_data) + (np.max(curt_data) - np.min(curt_data)) * np.random.rand()

            left_curt_index, right_curt_index = self.split_indices(curt_index, curt_data, self.split_point)

            self.left_child = SENCTree(data, left_curt_index, curt_height + 1, paras, all_train_data_label, globals_dict)
            if globals_dict['flag1'] == 1:
                self.center = np.mean(data[right_curt_index], axis=0)
                self.dist = np.max(cdist(data[right_curt_index], [self.center]))
                self.la = all_train_data_label[right_curt_index]
                globals_dict['flag1'] = 0

            self.right_child = SENCTree(data, right_curt_index, curt_height + 1, paras, all_train_data_label, globals_dict)
            if globals_dict['flag1'] == 1:
                self.center = np.mean(data[left_curt_index], axis=0)
                self.dist = np.max(cdist(data[left_curt_index], [self.center]))
                self.la = all_train_data_label[left_curt_index]
                globals_dict['flag1'] = 0
