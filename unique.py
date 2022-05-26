import argparse
import numpy as np
import pickle
from read import FEmapping

def show_result(name='fat.pkl'):
    with open(name, 'rb') as f:
        tst = pickle.load(f)
    for i in tst:
        print(i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--newfebfile_name', type=str, help='', default=r'breast_new.feb')
    args = parser.parse_args()
    febfile_name=args.newfebfile_name


    args = parser.parse_args()
    with open(febfile_name, "rb") as f:
        data = f.read()
    new_fe=FEmapping()
    Part_index = new_fe.get_Part_index(data,matname='fat')
    element = new_fe.get_Ele(data, Part_index)
    fat_8,fat_full=new_fe.get_node_single_ele(element)
    t=list(fat_8.values())
    k=list(fat_8.keys())
    remove_list=[]
    for i in range(len(t)):
        temp_pool=t[:i]+t[i+1:]
        flatten=np.array(temp_pool).flatten()
        nr=0
        for single in t[i]:
            if single in flatten:
                break
            else:
                nr+=1
        if nr == 8:
            print('id:',k[i],'should be alone element ')
            remove_list.append(k[i])
    tree = new_fe.read_xml('Breast06_py_new.feb')

    Elements = new_fe.find_nodes(tree, "Mesh/Elements")  # 找到Mesh节点
    Elements_fat = new_fe.get_node_by_keyvalue(Elements, {"type": "hex20", "name": "Part61"})
    Elements_tissue = new_fe.get_node_by_keyvalue(Elements, {"type": "hex20", "name": "Part62"})

    for i in remove_list:
        context = fat_full[i]
        context = ','.join(context)
        new_tissue = new_fe.create_node("elem", {"id": f"{i}"}, content=context)
        new_fe.del_node_by_tagkeyvalue(Elements_fat, "elem", {"id": f"{i}"})  # DEL NODE PART6

        new_fe.add_child_node(Elements_tissue, new_tissue)  # 插入到父节点之下
    new_feb_file='post_new'
    new_fe.write(tree, f"./{new_feb_file}.feb")
# show_result()