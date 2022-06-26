'''
install the below required third libs with pip install -r requirements.txt

And just run the read.py

In visualization part:

each single colourful point represents the center of each element

I think the result should be right :)

test
'''
import argparse
import pickle
from xml.etree.ElementTree import ElementTree, Element

import numpy as np
import pydicom
from bs4 import BeautifulSoup

from save import save_pickle
from visualization import show_result


class FEmapping():
    def __init__(self):
        self.fat_list = []
        self.tissue_list = []
        self.dicom_array = None
        self.Rows = None
        self.Columns = None
        self.PixelSpacing = []
        self.SliceThickness = []
        self.NumberofFrames = None
        self.fat_center = []
        self.tissue_center = []
        self.xyz_spacing = []
        self.breast_name = ''
        self.thresh = 200

    def get_Part_index(self, text, matname='breast'):
        soup = BeautifulSoup(text, 'xml')

        for link in soup.find_all('SolidDomain'):
            if link['mat'] == matname:
                print('In SolidDomain mat key is', link['name'])
                self.breast_name = link['name']
                return link['name']
        print('')
        return 'None'

    def get_Ele(self, text, Part_name):
        soup = BeautifulSoup(text, 'xml')

        for link in soup.find_all('Elements'):
            if link['name'] == f'{Part_name}':
                return link
            else:
                print(link['name'])
        print('Not found')
        return None

    def get_Eles(self, text, Part_name):
        soup = BeautifulSoup(text, 'xml')

        for link in soup.find_all('ElementSet'):
            if link['name'] == f'{Part_name}':
                return link
            else:
                print(link['name'])
        print('Not found')
        return None

    def get_id(self, data):
        id_set = []
        for e in data:
            if type(e) != type(data):
                continue
            id = e['id']
            id_set.append(id)
        return id_set

    def get_node_single_ele(self, data):
        element = {}
        element_full = {}
        for e in data:
            if type(e) != type(data):
                continue
            id = e['id']
            t = e.text.split(',')
            t = list(map(lambda x: x.strip(), t))
            element.setdefault(f'{id}', t[:8])
            element_full.setdefault(f'{id}', t)
        return element, element_full

    def get_node_dic(self, data, Breast_node_name=''):
        soup = BeautifulSoup(data, 'xml')
        node = {}
        for link in soup.find_all('Nodes'):
            try:
                if link['name'] == Breast_node_name:
                    for n in link:
                        if type(n) != type(link):
                            continue
                        id = n['id']
                        t = n.text.split(',')
                        node.setdefault(f'{id}', t)
            except KeyError:
                for n in link:
                    if type(n) != type(link):
                        continue
                    id = n['id']
                    t = n.text.split(',')
                    node.setdefault(f'{id}', t)
        with open("node.pkl", 'wb') as f:
            pickle.dump(node, f)
        return node

    def analyse(self, element_dic, node_dic):
        cord_node = []
        for k in node_dic.values():
            k = list(float(i) for i in k)
            cord_node.append(k)

        cloud = np.array(cord_node)
        max = np.max(cloud, axis=0)
        min = np.min(cloud, axis=0)
        # min[1:]=0
        xyz_range = max - min
        range = np.where(self.dicom_array > 0)
        x = np.max(range[0]) - np.min(range[0])
        y = np.max(range[1]) - np.min(range[1])
        z = np.max(range[2]) - np.min(range[2])
        # volumen_size=[self.Rows,self.Columns,self.NumberofFrames]
        volumen_size = [x, y, z]
        # self.xyz_spacing = list(map(lambda x: x[0] / x[1], zip(xyz_range,volumen_size)))
        self.xyz_spacing = [xyz_range[1] / volumen_size[0], xyz_range[0] / volumen_size[1],
                            xyz_range[2] / volumen_size[2]]
        for element_index in element_dic.items():
            cord = [0, 0, 0]
            for single_node in element_index[1]:
                temp = list(map(float, node_dic[single_node]))
                cord = list(map(lambda x: x[0] + x[1], zip(cord, temp)))
            # print(list(map(lambda x: x/8, cord)))
            center_cord = list(map(lambda x: x / 8, cord))
            result = self.FatOR_Tissue(center_cord)
            if result:
                self.Move_in_Fat(element_index[0])
                self.fat_center.append(cord)
            else:
                self.Move_in_Tissue(element_index[0])
                self.tissue_center.append(cord)

    def get_center_from_element(self, element_dic, node_dic, cls='Fat'):
        for element_index in element_dic.items():
            cord = [0, 0, 0]
            for single_node in element_index[1]:
                temp = list(map(float, node_dic[single_node]))
                cord = list(map(lambda x: x[0] + x[1], zip(cord, temp)))
            # print(list(map(lambda x: x/8, cord)))
            center_cord = list(map(lambda x: x / 8, cord))
            if cls == 'Fat':
                self.fat_center.append(cord)
            else:
                self.tissue_center.append(cord)



    def FatOR_Tissue(self, cord):
        thresh = self.thresh

        pixel_cord = list(map(lambda x: int(x[0] / x[1]), zip(cord, self.xyz_spacing)))
        try:
            c0 = min(pixel_cord[1] + self.Rows // 2, self.dicom_array.shape[0] - 1)
            c1 = min(pixel_cord[0] + self.Columns // 2, self.dicom_array.shape[1] - 1)
            c2 = min(pixel_cord[2] + self.NumberofFrames // 2, self.dicom_array.shape[2] - 1)
            # c0=min(pixel_cord[1]  -self.Rows,self.dicom_array.shape[0]-1)
            # c1=min(pixel_cord[0] -self.Columns,self.dicom_array.shape[1]-1)
            # c2=min(pixel_cord[2]-self.NumberofFrames//2,self.dicom_array.shape[2]-1)
            # print(c0,c1,c2)

            gray_value = self.dicom_array[c0, c1, c2]
        except:
            print(pixel_cord[1] + self.Rows // 2, pixel_cord[0] + self.Columns // 2,
                  pixel_cord[2] + self.NumberofFrames // 2)
            print(self.Rows, self.Columns, self.NumberofFrames)
        if gray_value > thresh:
            return True

        return False

    def Move_in_Fat(self, element_index):
        self.fat_list.append(element_index)

    def Move_in_Tissue(self, element_index):
        self.tissue_list.append(element_index)

    def read_xml(self, in_path):
        '''''读取并解析xml文件
           in_path: xml路径
           return: ElementTree'''
        tree = ElementTree()
        tree.parse(in_path)
        return tree

    def create_node(self, tag, property_map, content=''):
        '''新造一个节点
           tag:节点标签
           property_map:属性及属性值map
           content: 节点闭合标签里的文本内容
           return 新节点'''
        element = Element(tag, property_map)
        element.tail = '\n' + '\t' * 3
        if content == '':
            return element
        element.text = content
        return element

    def find_nodes(self, tree, path):
        '''''查找某个路径匹配的所有节点
           tree: xml树
           path: 节点路径'''
        return tree.findall(path)

    def add_child_node(self, nodelist, element):
        '''''给一个节点添加子节点
           nodelist: 节点列表
           element: 子节点'''
        for node in nodelist:
            node.append(element)
            # node.append('\n')

    def get_node_by_keyvalue(self, nodelist, kv_map):
        '''''根据属性及属性值定位符合的节点，返回节点
           nodelist: 节点列表
           kv_map: 匹配属性及属性值map'''

        def if_match(node, kv_map):
            '''''判断某个节点是否包含所有传入参数属性
               node: 节点
               kv_map: 属性及属性值组成的map'''
            for key in kv_map:
                if node.get(key) != kv_map.get(key):
                    return False
            return True

        result_nodes = []
        for node in nodelist:
            if if_match(node, kv_map):
                result_nodes.append(node)
        return result_nodes

    def del_node_by_tagkeyvalue(self, nodelist, tag, kv_map):
        '''''同过属性及属性值定位一个节点，并删除之
           nodelist: 父节点列表
           tag:子节点标签
           kv_map: 属性及属性值列表'''

        def if_match(node, kv_map):
            '''''判断某个节点是否包含所有传入参数属性
               node: 节点
               kv_map: 属性及属性值组成的map'''
            for key in kv_map:
                if node.get(key) != kv_map.get(key):
                    return False
            return True

        for parent_node in nodelist:
            children = list(parent_node)
            for child in children:
                if child.tag == tag and if_match(child, kv_map):
                    parent_node.remove(child)

    def read_dicom(self, dicom):
        head = pydicom.read_file(dicom)
        dicom_volumen = pydicom.dcmread(dicom)

        self.dicom_array = dicom_volumen.pixel_array
        self.dicom_array = np.moveaxis(self.dicom_array, 0, 2)
        self.PixelSpacing = head[0x5200, 0x9230][0]['PixelMeasuresSequence'][0]['PixelSpacing'].value
        self.SliceThickness = float([head[0x5200, 0x9230][0]['PixelMeasuresSequence'][0]['SliceThickness'].value][0])
        self.NumberofFrames = head[0x0028, 0x0008].value
        self.Rows = head[0x0028, 0x0010].value
        self.Columns = head[0x0028, 0x0011].value
        # self.dicom_array=(self.dicom_array/np.max(self.dicom_array))*255

    def get_info(self, Name):
        try:
            info = self.__getattribute__(Name)
        except:
            raise AttributeError(f'No attribute named :{Name}')

        return info

    def write(self, tree, out_path):
        '''''将xml文件写出
               tree: xml树
               out_path: 写出路径'''
        tree.write(out_path, encoding="ISO-8859-1", xml_declaration=True)

    def get_result(self):
        return self.fat_list, self.tissue_list

    def get_center(self):
        return self.fat_center, self.tissue_center


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--febfile_name', type=str, help='', default=r'breast.feb')
    parser.add_argument('--dcm_name', type=str, help='', default='breast.dcm')
    parser.add_argument('--Node_name', type=str, help='', default='breast')
    parser.add_argument('--mat_name', type=str, help='', default='matbreast')

    args = parser.parse_args()
    print(args)

    # step 1 : initialize all name parameters
    febfile_name = args.febfile_name
    dcm_name = args.dcm_name
    Node_name = args.Node_name
    mat_name = args.mat_name

    # step 2 : open the dcm and get infos from it

    with open(febfile_name, "rb") as f:
        feb_data = f.read()
    fe = FEmapping()
    fe.read_dicom(dcm_name)

    # step 3 : analyse feb file
    '''
    workflow :
    according to matname in  'MeshDomains - SolidDomain' to know the element name ,i.e 'Part1'
    according to element name to get element infos
    according to args.Node_name to get node infos
    
    '''

    Part = fe.get_Part_index(feb_data, matname=mat_name)
    element = fe.get_Ele(feb_data, Part)
    element_dic, element_dic_full = fe.get_node_single_ele(element)
    node_dic = fe.get_node_dic(feb_data, f'{Node_name}')
    fe.analyse(element_dic, node_dic)
    fat_list, tissue_list = fe.get_result()
    fat_center, tissue_center = fe.get_center()
    with open("fat.pkl", 'wb') as f:
        pickle.dump(fat_center, f)
    with open(f"tissue.pkl", 'wb') as f:
        pickle.dump(tissue_center, f)

    save_pickle(fat_list, node_dic, element_dic_full, 'fat_node')
    save_pickle(tissue_list, node_dic, element_dic_full, 'tissue_node')

    show_result("fat.pkl", vox=True)
    show_result("tissue.pkl")
    # step 4 : modify feb file

    tree = fe.read_xml(febfile_name)
    Original_name = fe.breast_name

    Elements_Fat = fe.create_node("Elements", {"type": "hex20", "name": "Part61"})  # 新建Fat节点
    Elements_Tissue = fe.create_node("Elements", {"type": "hex20", "name": "Part62"})  # 新建Tissue节点

    MeshDomains_Fat = fe.create_node("SolidDomain", {"name": "Part61", "mat": "fat"})  # 新建Tissue节点
    MeshDomains_Tissue = fe.create_node("SolidDomain", {"name": "Part62", "mat": "Tissue"})  # 新建Tissue节点

    Mesh_nodes = fe.find_nodes(tree, "Mesh")  # 找到Mesh节点
    MeshDomains = fe.find_nodes(tree, "MeshDomains")  # 找到Mesh节点
    fe.del_node_by_tagkeyvalue(Mesh_nodes, "Elements", {"name": f"{Original_name}"})  # DEL NODE PART6
    fe.add_child_node(Mesh_nodes, Elements_Fat)  # 插入到父节点之下
    fe.add_child_node(Mesh_nodes, Elements_Tissue)  # 插入到父节点之下
    fe.add_child_node(MeshDomains, MeshDomains_Fat)  # 插入到父节点之下
    fe.add_child_node(MeshDomains, MeshDomains_Tissue)  # 插入到父节点之下

    # step 5 : rewrite feb file

    Mesh_nodes = fe.find_nodes(tree, "Mesh/Elements")
    Elements_Fat = fe.get_node_by_keyvalue(Mesh_nodes, {"type": "hex20", "name": "Part61"})  # 通过属性准确定位子节点
    Elements_Tissue = fe.get_node_by_keyvalue(Mesh_nodes, {"type": "hex20", "name": "Part62"})  # 通过属性准确定位子节点

    for index in fat_list:
        context = element_dic_full[index]
        context = ','.join(context)
        # context = ','.join(test[1])
        new_fat = fe.create_node("elem", {"id": f"{index}"}, content=context)

        fe.add_child_node(Elements_Fat, new_fat)  # 插入到父节点之下
    for index in tissue_list:
        context = element_dic_full[index]
        context = ','.join(context)
        new_tissue = fe.create_node("elem", {"id": f"{index}"}, content=context)

        fe.add_child_node(Elements_Tissue, new_tissue)  # 插入到父节点之下

    # step 6 : save new feb file

    new_feb_file = Node_name + '_new'
    fe.write(tree, f"./{new_feb_file}.feb")
