import  numpy as np
import os
import xml.etree.ElementTree as ET


if __name__ == '__main__':
     path='./img_randomdrop_nutAndscrew/'
     listdir=os.listdir(path)
     for i in range(len(listdir)):
         if listdir[i].find(".png")!= -1:
              newname=listdir[i].replace(".png",'.jpg')
              os.rename(path+listdir[i],path+newname)

          # in_file = open(path_xml+ listdir[i],encoding='utf-8')
          # tree = ET.parse(in_file)
          # root = tree.getroot()
          # change = 0
          # filename= tree.find('filename').text
          # if filename.find('.png')!=-1:
          #     newfilename=filename.replace('.png','.jpg')
          #     newfilename = newfilename.replace('微信图片', '1')
          #     tree.find('filename').text=newfilename
          #     change = 1
          # path = tree.find('path').text
          # if path.find('.png')!=-1:
          #     newpath=path.replace('.png','.jpg')
          #     newpath = newpath.replace('微信图片', '1')
          #     tree.find('path').text=newpath
          #     change = 1
          #
          # # for obj in root.iter('object'):
          # #      difficult = obj.find('difficult').text
          # #      cls = obj.find('name').text
          # if change == 1:
          #      tree.write(path_xml+listdir[i])
