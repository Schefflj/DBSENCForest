import numpy as np
from scipy.spatial.distance import cdist
from time import time
from DBSENCForest.data_structure.senc_forest.senc_tree2 import SENCTree2


class SencForestsAlgorithm:

    # Global variables
    global_vars = {
        'id2': 0,
        'pathlinenew': [],
        'pathline4': [],
        'pathline5': [],
        'flag1': 0,
        'id': 0,
        'pathline': [],
        'pathline3': np.empty((0, 1))
    }

    # Define the tabulate function to mimic MATLAB's tabulate
    def tabulate(self, arr):
        """
        Mimics MATLAB's tabulate function.

        :param arr: numpy array, input array to tabulate.
        :return: numpy array, unique values and their counts.
        """
        vals, counts = np.unique(arr, return_counts=True)
        return np.column_stack((vals, counts))

    # Global flag
    flag = 0

    def sence_mass(self, data, curt_index, tree, mass, cldi, trave, ano, globals_dict):
        """
        Recursively calculates mass for a given data point in the SENC tree.

        :param data: numpy array, input data.
        :param curt_index: numpy array, current indices being processed.
        :param tree: SENCTree, the current tree node.
        :param mass: numpy array, mass array to store results.
        :param cldi: float, distance multiplier.
        :param trave: numpy array, traversal array.
        :param ano: float, anomaly threshold.
        :param globals_dict: dict, global variables dictionary.
        :return: numpy array, updated mass array.
        """
        global flag
        flag = 0

        if tree.node_status == 0:
            mass[curt_index, 0] = float(tree.high) < ano
            if tree.size == 1:
                mass[curt_index, 2] = tree.la
                mass[curt_index, 3] = tree.id
                flag = 1
            elif tree.size < 1:
                flag = 1
                mass[curt_index, 3] = tree.id
            else:
                tempdist = cdist(data[curt_index], [tree.center])
                mass[curt_index, 1] = tempdist > tree.dist * cldi
                mass[curt_index, 3] = tree.id

                ter = tree.la
                scoretrainl = self.tabulate(ter)
                if scoretrainl.size == 0:
                    mass[curt_index, 2] = -1
                else:
                    scoretrainl = scoretrainl[scoretrainl[:, 1] == np.max(scoretrainl[:, 1]), 0]
                    if len(scoretrainl) > 1:
                        mass[curt_index, 2] = scoretrainl[0]
                    else:
                        mass[curt_index, 2] = scoretrainl[0]

                if mass[curt_index, 1] == 1 and mass[curt_index, 0] == 1:
                    mass[curt_index, 4] = 1
                else:
                    mass[curt_index, 4] = 0
            return mass
        else:
            left_curt_index = curt_index[data[curt_index, tree.split_attribute] < tree.split_point]
            right_curt_index = np.setdiff1d(curt_index, left_curt_index)
            trave[0, tree.split_attribute] = 1
            trave[1, tree.split_attribute] = tree.split_point
            if len(left_curt_index) > 0:
                mass = self.sence_mass(data, left_curt_index, tree.left_child, mass, cldi, trave, ano, globals_dict)
                if flag == 1:
                    tempdist = cdist(data[curt_index], [tree.center])
                    mass[curt_index, 1] = tempdist > tree.dist * cldi

                    if mass[curt_index, 0] < ano and mass[curt_index, 1] == 1:
                        mass[curt_index, 4] = 1
                    else:
                        mass[curt_index, 4] = 0

                    ter = tree.la[right_curt_index]
                    scoretrainl = self.tabulate(ter)
                    if scoretrainl.size == 0:
                        mass[curt_index, 2] = -1
                    else:
                        scoretrainl = scoretrainl[scoretrainl[:, 1] == np.max(scoretrainl[:, 1]), 0]
                        if len(scoretrainl) > 1:
                            mass[curt_index, 2] = scoretrainl[0]
                        else:
                            mass[curt_index, 2] = scoretrainl[0]
                    flag = 0
            if len(right_curt_index) > 0:
                mass = self.sence_mass(data, right_curt_index, tree.right_child, mass, cldi, trave, ano, globals_dict)
                if flag == 1:
                    tempdist = cdist(data[curt_index], [tree.center])
                    mass[curt_index, 1] = tempdist > tree.dist * cldi

                    if mass[curt_index, 0] < ano and mass[curt_index, 1] == 1:
                        mass[curt_index, 4] = 1
                    else:
                        mass[curt_index, 4] = 0

                    ter = tree.la[left_curt_index]
                    scoretrainl = self.tabulate(ter)
                    if scoretrainl.size == 0:
                        mass[curt_index, 2] = -1
                    else:
                        scoretrainl = scoretrainl[scoretrainl[:, 1] == np.max(scoretrainl[:, 1]), 0]
                        if len(scoretrainl) > 1:
                            mass[curt_index, 2] = scoretrainl[0]
                        else:
                            mass[curt_index, 2] = scoretrainl[0]
                    flag = 0
        return mass

    def sence_estimation(self, test_data, forest, cldi, anomalylambdan):
        """
        Estimates the mass for a given test dataset using the SENC forest.

        :param test_data: numpy array, data to be tested.
        :param forest: SENCForest, the forest containing the trees.
        :param cldi: float, distance multiplier.
        :param anomalylambdan: numpy array, anomaly thresholds.
        :return: tuple, containing the mass array and the elapsed time.
        """
        num_inst = test_data.shape[0]
        mass = np.zeros((forest.num_tree, 5))
        start_time = time()

        for k in range(forest.num_tree):
            trave = np.zeros((2, test_data.shape[1]))
            ano = anomalylambdan[k]
            mass[k, :] = self.sence_mass(test_data, np.arange(num_inst), forest.trees[k], np.zeros((num_inst, 5)), cldi,
                                         trave, ano, {'id': id})

        elapse_time = time() - start_time
        return mass, elapse_time

    def updatetree(self, temptree, idtree, alltraindata, alltraindatalabel, newlabel, global_vars):
        """
        Updates a tree based on new training data.

        :param temptree: SENCTree, the tree to update.
        :param idtree: numpy array, IDs associated with the tree.
        :param alltraindata: numpy array, all training data.
        :param alltraindatalabel: numpy array, labels for the training data.
        :param newlabel: numpy array, new labels for updating.
        :param global_vars: dict, global variables.
        :return: SENCTree, the updated tree.
        """
        if temptree.node_status == 0:
            tempid = np.where(idtree == temptree.id)[0]
            if len(tempid) > 0:
                global_vars['pathline3'] = np.empty((0, 1))
                updata = alltraindata[tempid, :]
                updatalabel = alltraindatalabel[tempid, :]
                if temptree.size != 0:
                    updata = np.vstack([updata, np.tile(temptree.center, (temptree.size, 1))])
                    updatalabel = np.hstack([updatalabel.flatten(), temptree.la])

                index_dim = np.arange(updata.shape[1])
                paras = {
                    'NumDim': updata.shape[1],
                    'HeightLimit': 50,
                    'IndexDim': index_dim
                }

                index_sub = np.arange(updata.shape[0])
                global_vars['rt'] = temptree.height + 3
                newtree = SENCTree2(updata, index_sub, temptree.height, paras, updatalabel, newlabel)
            else:
                newtree = temptree
                if len(temptree.la) > 0:
                    global_vars['pathline5'].append(temptree.high)
            return newtree
        else:
            left_child = temptree.left_child
            right_child = temptree.right_child

            temptree.left_child = self.updatetree(left_child, idtree, alltraindata, alltraindatalabel, newlabel, global_vars)
            temptree.right_child = self.updatetree(right_child, idtree, alltraindata, alltraindatalabel, newlabel,
                                              global_vars)

        return temptree

    def update_model(self, all_train_data, re_forest, all_train_data_label, id_buffer, num_sub):
        """
        Updates the model by modifying the forest with new data.

        :param all_train_data: numpy array, all training data.
        :param re_forest: SENCForest, the forest to be updated.
        :param all_train_data_label: numpy array, labels for the training data.
        :param id_buffer: numpy array, buffer of IDs.
        :param num_sub: int, number of subsamples to use.
        :return: SENCForest, the updated forest.
        """
        self.global_vars['id2'] = 1

        select_number = num_sub

        for i in range(re_forest.num_tree):
            self.global_vars['pathline4'] = []
            id_tree = id_buffer[i % len(id_buffer), :]
            temp_tree = re_forest.trees[i]
            self.global_vars['id2'] = temp_tree.totalid + 1
            self.global_vars['pathlinenew'] = []
            new_label = np.unique(all_train_data_label)
            self.global_vars['pathline5'] = []

            indices = np.random.permutation(all_train_data.shape[0])
            id_tree1 = id_tree[indices[:select_number]]
            all_train_data1 = all_train_data[indices[:select_number], :]
            all_train_data_label1 = all_train_data_label[indices[:select_number], :]

            new_tree = self.updatetree(temp_tree, id_tree1, all_train_data1, all_train_data_label1, new_label, self.global_vars)

            new_tree.totalid = self.global_vars['id2'] - 1
            new_tree.pathline1 = self.global_vars['pathline5']
            temp_an = np.sort(self.global_vars['pathline5'])

            vars_rate2 = []
            for j in range(1, len(temp_an)):
                varsf = np.std(temp_an[:j])
                varsb = np.std(temp_an[j:])
                vars_rate2.append(abs(varsf - varsb))

            bb = np.argmin(vars_rate2)
            print(f'old para is {re_forest.anomaly[i]}; new para is {temp_an[bb]}.')

            re_forest.anomaly[i] = temp_an[bb]
            re_forest.trees[i] = new_tree

        return re_forest

    def testing_pro(self, stream_data, stream_data_label, model, para):
        """
        Processes streaming data and tests against the model.

        :param stream_data: numpy array, data stream to be tested.
        :param stream_data_label: numpy array, true labels of the stream data.
        :param model: SENCForest, the trained forest model.
        :param para: dict, parameters including buffer size and alpha.
        :return: numpy array, results with predicted and true labels.
        """
        new_class_label = 4
        buffer = []
        result_new = []
        id_buffer = []
        batch_data_label = []
        batch_data_label_true = []

        for j in range(stream_data.shape[0]):
            mass, mtimetest = self.sence_estimation(stream_data[j, :].reshape(1, -1), model, para['alpha'],
                                                    model.anomaly)



            answermass = mass[:, 2].copy()
            answermass[mass[:, 4] == 1] = new_class_label
            score = self.tabulate(answermass)
            score_1 = score[score[:, 1] == np.max(score[:, 1]), :]

            if score_1.size == 0 or score_1[0, 0] == new_class_label:
                buffer.append(stream_data[j, :])
                id_buffer.append(mass[:, 3])
                batch_data_label.append(new_class_label)
                batch_data_label_true.append(stream_data_label[j])
                result_new.append([new_class_label, stream_data_label[j]])
            else:
                result_new.append([score_1[0, 0], stream_data_label[j]])

            if len(buffer) >= para['buffersize']:
                model = self.update_model(np.array(buffer), model, np.array(batch_data_label).reshape(-1, 1),
                                          np.array(id_buffer))
                buffer = []
                id_buffer = []
                batch_data_label = []

            print(f'{j + 1}')

        return np.array(result_new, dtype=object)