

'''
install the below required third libs with pip install -r requirements.txt

And just run the read.py

In visualization part:

each single colourful point represents the center of each element

I think the result should be right :)

test
'''
import pickle
from save import save_pickle
from bs4 import BeautifulSoup
from xml.etree.ElementTree import ElementTree, Element
import numpy as np
from visualization import show_result
import pydicom


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
        self.fat_center=[]
        self.tissue_center=[]
        self.xyz_spacing =[]
    def get_Part_index(self, text):
        soup = BeautifulSoup(text, 'xml')

        for link in soup.find_all('SolidDomain'):
            if link['mat'] == 'breast':
                print('In SolidDomain mat key is',link['name'])
                return link['name']
        return 'None'

    def get_Ele(self, text, Part):
        soup = BeautifulSoup(text, 'xml')

        for link in soup.find_all('Elements'):

            if link['name'] == f'{Part}':
                return link

    def get_node_single_ele(self, data):
        element = {}
        for e in data:
            if type(e) != type(data):
                continue
            id = e['id']
            t = e.text.split(',')
            element.setdefault(f'{id}', t[:8])
        return element

    def get_node_dic(self, data, Breast_node_name='breast06_LCC'):
        soup = BeautifulSoup(data, 'xml')
        node = {}
        for link in soup.find_all('Nodes'):
            if link['name'] == Breast_node_name:
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
        cord_node=[]
        for k in node_dic.values():
            k = list(float(i) for i in k)
            cord_node.append(k)

        cloud = np.array(cord_node)
        max=np.max(cloud,axis=0)
        min=np.min(cloud,axis=0)
        # min[1:]=0
        xyz_range=max-min

        volumen_size=[self.Rows,self.Columns,self.NumberofFrames]
        # self.xyz_spacing = list(map(lambda x: x[0] / x[1], zip(xyz_range,volumen_size)))
        self.xyz_spacing = [xyz_range[1]/volumen_size[0],xyz_range[0]/volumen_size[1],xyz_range[2]/volumen_size[2]]
        for element_index in element_dic.items():
            cord = [0, 0, 0]
            for single_node in element_index[1]:
                temp = list(map(float, node_dic[single_node]))
                cord = list(map(lambda x: x[0] + x[1], zip(cord, temp)))
            # print(cord)
            # print(list(map(lambda x: x/8, cord)))
            center_cord = list(map(lambda x: x / 8, cord))
            result = self.FatOR_Tissue(center_cord)
            if result:
                self.Move_in_Fat(element_index[0])
                self.fat_center.append(cord)
            else:
                self.Move_in_Tissue(element_index[0])
                self.tissue_center.append(cord)


    def FatOR_Tissue(self, cord, thresh=200):
        offset=[0,self.Columns//2,0]

        pixel_cord = list(map(lambda x: int(x[0] / x[1]), zip(cord,self.xyz_spacing )))
        try:
            c0=min(pixel_cord[1]  +self.Rows//2,self.dicom_array.shape[0]-1)
            c1=min(pixel_cord[0] +self.Columns//2,self.dicom_array.shape[1]-1)
            c2=min(pixel_cord[2]+self.NumberofFrames//2,self.dicom_array.shape[2]-1)


            gray_value=self.dicom_array[c0,c1,c2 ]
        except:
            print(pixel_cord[1] +self.Rows//2 ,pixel_cord[0] +self.Columns//2,pixel_cord[2]+self.NumberofFrames//2)
            print(self.Rows,self.Columns,self.NumberofFrames)
        if gray_value>thresh:

            return  True
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
        element.tail='\n'+'\t'*3
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
        self.dicom_array=np.moveaxis(self.dicom_array,0,2)
        self.PixelSpacing = head[0x5200, 0x9230][0]['PixelMeasuresSequence'][0]['PixelSpacing'].value
        self.SliceThickness = [head[0x5200, 0x9230][0]['PixelMeasuresSequence'][0]['SliceThickness'].value]

        self.NumberofFrames = head[0x0028, 0x0008].value
        self.Rows = head[0x0028, 0x0010].value
        self.Columns = head[0x0028, 0x0011].value


    def write(self, tree, out_path):
        '''''将xml文件写出
               tree: xml树
               out_path: 写出路径'''
        tree.write(out_path, encoding="ISO-8859-1", xml_declaration=True)
    def get_result(self):
        return self.fat_list,self.tissue_list

    def get_center(self):
        return self.fat_center,self.tissue_center

if __name__ == '__main__':
    with open("breast.feb", "rb") as f:
        data = f.read()
    fe = FEmapping()
    fe.read_dicom('Breast06_py.dcm')
    Part = fe.get_Part_index(data)
    element = fe.get_Ele(data, Part)

    element_dic = fe.get_node_single_ele(element)
    node_dic = fe.get_node_dic(data,'Breast06_py')
    fe.analyse(element_dic, node_dic)
    fat_list, tissue_list=fe.get_result()
    a, b=fe.get_center()
    with open("fat.pkl", 'wb') as f:
        pickle.dump(a, f)
    with open(f"tissue.pkl", 'wb') as f:
        pickle.dump(b, f)
    save_pickle(fat_list,node_dic,element_dic, 'fat_node')
    save_pickle(tissue_list,node_dic,element_dic, 'tissue_node')
    show_result("fat.pkl")
    show_result("tissue.pkl")

#TODO: modifiy the feb file but not yet
    tree = fe.read_xml("breast.feb")
    Elements_Fat = fe.create_node("Elements", {"type": "hex20", "name": "Part61"})  # 新建Fat节点
    Elements_Tissue = fe.create_node("Elements", {"type": "hex20", "name": "Part62"})  # 新建Tissue节点
    Mesh_nodes = fe.find_nodes(tree, "Mesh")  # 找到Mesh节点

    # origin_breast_nodes = fe.get_node_by_keyvalue(Elements_nodes, {"name": "Part6"}) # 通过属性准确定位子节点
    target_del_node = fe.del_node_by_tagkeyvalue(Mesh_nodes, "Elements", {"name": "Part6"})  # DEL NODE PART6
    # Elements_Fat=fe.get_node_by_keyvalue(Elements_nodes, {"name": "Part61"}) # 通过属性准确定位子节点
    fe.add_child_node(Mesh_nodes, Elements_Fat) # 插入到父节点之下
    Mesh_nodes = fe.find_nodes(tree, "Mesh/Elements")  # 找到Mesh_nodes节点
    Elements_Fat = fe.get_node_by_keyvalue(Mesh_nodes, {"type": "hex20", "name": "Part61"})  # 通过属性准确定位子节点

    for index in fat_list:
        context=element_dic[index]
        context=','.join(context)
        # context = ','.join(test[1])
        new_fat = fe.create_node("elem", {"id": f"{index}"}, content=context)

        fe.add_child_node(Elements_Fat, new_fat)  # 插入到父节点之下

    fe.write(tree, "./xiugai.feb")
