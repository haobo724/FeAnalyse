import argparse
import numpy as np
import pickle
from read import FEmapping

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--newfebfile_name', type=str, help='', default=r'F:\Siemens\pachong\526\breast03_15.feb')
    args = parser.parse_args()
    febfile_name = args.newfebfile_name

    args = parser.parse_args()
    with open(febfile_name, "rb") as f:
        data = f.read()
    new_fe = FEmapping()

    Part61 = new_fe.get_Ele(data, Part_name='Part61')
    Part62 = new_fe.get_Ele(data, Part_name='Part62')
    SurfFat = new_fe.get_Eles(data, Part_name='SurfFat')

    elements_full_61 = new_fe.get_node_single_ele(Part61)[1]
    elements_full_62 = new_fe.get_node_single_ele(Part62)[1]
    elements_full_SurfFat = new_fe.get_node_single_ele(SurfFat)[1]

    id_Part61 = np.array(list(elements_full_61.keys()))
    id_Part62 = np.array(list(elements_full_62.keys()))
    id_SurfFat = np.array(list(elements_full_SurfFat.keys()))

    intersection_61_surfFat = np.intersect1d(id_Part61, id_SurfFat)
    # print("\nThe intersection between the two arrays is:\n", intersection_61_surfFat)
    print("\nThe intersection between Part61 and id_SurfFat :\n", len(intersection_61_surfFat))
    intersection_62_surfFat = np.intersect1d(id_Part62, id_SurfFat)
    print("\nThe intersection between id_Part62 and id_SurfFat:\n", len(intersection_62_surfFat))

    intersection = np.intersect1d(intersection_61_surfFat, intersection_62_surfFat)
    print("\nThe intersection between intersection_61_surfFat and intersection_62_surfFat:\n", len(intersection))
    assert len(intersection) ==

    tree = new_fe.read_xml(febfile_name)

    Elements = new_fe.find_nodes(tree, "Mesh/Elements")  # 找到Mesh节点
    Elements_fat = new_fe.get_node_by_keyvalue(Elements, {"type": "hex20", "name": "Part61"})
    Elements_tissue = new_fe.get_node_by_keyvalue(Elements, {"type": "hex20", "name": "Part62"})

    for i in intersection_62_surfFat:
        context = elements_full_62[i]
        context = ','.join(context)
        new_node = new_fe.create_node("elem", {"id": f"{i}"}, content=context)
        new_fe.del_node_by_tagkeyvalue(Elements_tissue, "elem", {"id": f"{i}"}) #del
        new_fe.add_child_node(Elements_fat, new_node)  # add

    new_feb_file = 'post_new'
    new_fe.write(tree, f"./{new_feb_file}.feb")
