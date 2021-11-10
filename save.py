import pickle
import numpy as np

def save_pickle(index_list,node_dic,element_dic,name):
    only_ele = []

    for tissue_index in index_list:
        try:
            only_ele.append(element_dic[tissue_index])
        except:
            continue

    temp = np.array(only_ele).flatten()
    temp = np.unique(temp)
    # print(len(temp))
    # print(len(np.unique(temp)))
    new_node = []
    for s in temp:

        # if node_dic[s] not in new_node:
        new_node.append(node_dic[s])
    with open(f"{name}.pkl", 'wb') as f:
        pickle.dump(new_node, f)

