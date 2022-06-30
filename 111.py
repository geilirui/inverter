import  keras_preprocessing.image as ksimg
import os
lst=os.listdir('./VOCdevkit/VOC2007/convert/')
for i in range(len(lst)):
    if lst[i].find('.png')!=-1:
        img=ksimg.load_img('./VOCdevkit/VOC2007/convert/'+lst[i])
        ksimg.save_img('./VOCdevkit/VOC2007/convert/'+lst[i].split('.')[0]+'.jpg',img)

