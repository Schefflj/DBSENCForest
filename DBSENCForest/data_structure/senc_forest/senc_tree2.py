import numpy as np
from scipy.spatial.distance import cdist

global_vars = {
    'flag1': 0,
    'id2': 1,
    'pathline3': [],
    'pathlinenew': [],
    'pathline5': [],
    'rt': 0
}


class SENCTree2:
    def __init__(self, data, curt_index, curt_height, paras, all_train_data_label, new_label):
        """
        Initializes the SENCTree2 object and constructs the tree.

        :param data: numpy array, input data used to build the tree.
        :param curt_index: numpy array, current indices of the data points being processed.
        :param curt_height: int, current height of the node in the tree.
        :param paras: dict, parameters for tree construction (e.g., height limit, index dimensions).
        :param all_train_data_label: numpy array, labels for the training data.
        :param new_label: numpy array, new labels used for updating the tree.
        """
        self.height = curt_height
        self.size = len(curt_index)
        self.curt_index = curt_index
        self.node_status = None
        self.split_attribute = None
        self.split_point = None
        self.left_child = None
        self.right_child = None
        self.center = None
        self.dist = None
        self.la = None
        self.id = None
        self.high = None
        self.left_curt_index = None
        self.left_curt_index_la = None
        self.right_curt_index = None
        self.right_curt_index_la = None

        self.build_tree(data, curt_index, curt_height, paras, all_train_data_label, new_label)

    def build_tree(self, data, curt_index, curt_height, paras, all_train_data_label, new_label):
        """
        Recursively builds the tree by splitting the data into nodes.

        :param data: numpy array, input data used to build the tree.
        :param curt_index: numpy array, current indices of the data points being processed.
        :param curt_height: int, current height of the node in the tree.
        :param paras: dict, parameters for tree construction (e.g., height limit, index dimensions).
        :param all_train_data_label: numpy array, labels for the training data.
        :param new_label: numpy array, new labels used for updating the tree.
        """
        num_inst = len(curt_index)

        if num_inst <= 10 or curt_height >= global_vars['rt']:
            self.node_status = 0
            self.split_attribute = None
            self.split_point = None
            self.left_child = None
            self.right_child = None
            self.size = num_inst
            self.curt_index = curt_index
            self.la = all_train_data_label[curt_index]

            self.id = global_vars['id2']
            global_vars['id2'] += 1
            C = np.mean(data[curt_index], axis=0)
            self.dist = np.max(cdist(data[curt_index], [C]))
            self.center = C

            if num_inst != 1:
                c = 2 * (np.log(self.size - 1) + 0.5772156649) - 2 * (self.size - 1) / self.size
            else:
                c = 0
            self.high = curt_height + c
            global_vars['pathline3'] = np.append(global_vars['pathline3'], [self.high] * num_inst)
            global_vars['pathline5'].append(self.high)

            if num_inst == 1:
                global_vars['flag1'] = 1

            return

        self.node_status = 1
        rindex = np.random.choice(paras['IndexDim'])
        self.split_attribute = rindex
        curt_data = data[curt_index, self.split_attribute]
        self.split_point = np.min(curt_data) + (np.max(curt_data) - np.min(curt_data)) * np.random.rand()

        left_curt_index = curt_index[curt_data < self.split_point]
        right_curt_index = np.setdiff1d(curt_index, left_curt_index)

        self.left_curt_index = left_curt_index
        self.left_curt_index_la = all_train_data_label[left_curt_index]
        self.right_curt_index = right_curt_index
        self.right_curt_index_la = all_train_data_label[right_curt_index]
        self.size = num_inst

        self.left_child = SENCTree2(data, left_curt_index, curt_height + 1, paras, all_train_data_label, new_label)
        if global_vars['flag1'] == 1:
            C = np.mean(data[right_curt_index], axis=0)
            self.dist = np.max(cdist(data[right_curt_index], [C]))
            self.center = C
            global_vars['flag1'] = 0
            if num_inst != 1:
                c = 2 * (np.log(self.size - 1) + 0.5772156649) - 2 * (self.size - 1) / self.size
            else:
                c = 0
            self.high = curt_height + c

        self.right_child = SENCTree2(data, right_curt_index, curt_height + 1, paras, all_train_data_label, new_label)
        if global_vars['flag1'] == 1:
            C = np.mean(data[left_curt_index], axis=0)
            self.dist = np.max(cdist(data[left_curt_index], [C]))
            self.center = C
            global_vars['flag1'] = 0
            if num_inst != 1:
                c = 2 * (np.log(self.size - 1) + 0.5772156649) - 2 * (self.size - 1) / self.size
            else:
                c = 0
            self.high = curt_height + c