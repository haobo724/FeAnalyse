import pickle
import numpy as np

def save_pickle(index_list,node_dic,element_dic,name):
    only_fat_ele = []
    only_tissue_ele = []


    for tissue in index_list:
        only_tissue_ele.append(element_dic[tissue])

    temp = np.array(only_fat_ele).flatten()
    new_node = []
    for s in temp:
        if node_dic[s] not in new_node:
            new_node.append(node_dic[s])

    with open(f"{name}.pkl", 'wb') as f:
        pickle.dump(new_node, f)

