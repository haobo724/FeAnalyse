import argparse

from bs4 import BeautifulSoup, NavigableString

from read import FEmapping
class new_FE(FEmapping):
    def get_all_Ele(self, text):
        soup = BeautifulSoup(text, 'xml')
        elements=[]
        for link in soup.find_all('Elements'):
            elements.append(link)

        return elements

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--febfile_name', type=str, help='', default=r'mat_fat_r.feb')
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
    new = new_FE()

    with open(febfile_name, "rb") as f:
        feb_data = f.read()
    fe = FEmapping()
    txtname = 'id_elements.txt'
    # txt_name_list = []
    ids = []
    elements_id = []
    elements = new.get_all_Ele(feb_data)
    print(len(elements),'Element(s) were found, they are:')
    for idx,i in enumerate(elements):
        print(idx+1,':',i['name'])
        # txt_name_list.append(i['name'])
        # print(i['id'])
        for elem in i:
            # if type(elem) != type(i):
            #     print(type(elem),type(i))
            #     continue
            if isinstance(elem, NavigableString):
                continue
            id = elem['id']
            ids.append(id)
        elements_id.append(ids)
        ids=[]
    i = 1
    with open(txtname, "w") as f:

        for ids in elements_id:
            for id in ids:
                content = id+','+str(i) +'\n'
                f.writelines(content)
            i+=1