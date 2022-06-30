import xml.etree.ElementTree as ET
import  os


sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ['nut_b','screw_s']
#classes = ['drop_nut','drop_screw']

def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    # xml_text = in_file.read()
    # tree = ET.fromstring(xml_text)
    change = 0
    # filename= tree.find('filename').text
    # if filename.find('.png')!=-1:
    #     newfilename=filename.replace('.png','.jpg')
    #     tree.find('filename').text=newfilename
    #     change = 1
    # path = tree.find('path').text
    # if path.find('.png')!=-1:
    #     newpath=path.replace('.png','.jpg')
    #     tree.find('filename').text = newpath
    #     change = 1

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # if cls == 'not drop':
        #     obj.find('name').text = 'notdrop'
        #     change = 1
        # if cls == 'ok':
        #     obj.find('name').text = 'OK'
        #     change = 1
        if cls == 'drop_srew':
            obj.find('name').text = 'drop_screw'
            change = 1
        # if cls == 'NOT DROP':
        #     obj.find('name').text = 'notdrop'
        #     change = 1
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
    if change==1:
        tree.write('%s/VOCdevkit/VOC%s/Annotations/%s.xml'%(wd,year, image_id))

wd = os.getcwd()

for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/Annotations/%s.jpg'%(wd, year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()
