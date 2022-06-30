import cv2
import numpy as np
from keras.preprocessing import image as ksimg
import os
import time
from yolo import YOLO
yolo = YOLO()
img_dir='./img/'
img_lst=os.listdir(img_dir)
import matplotlib.pyplot as plt
image_path='./img/P1546778-84-E-SJBL22149ADE060.png'
template_path='template.jpg'
for i in range(len(img_lst)):
    template=cv2.imread(template_path,0)
    cv2_img_gray=cv2.imread(img_dir+img_lst[i],0)
    cv2_img_RGB=ksimg.load_img(img_dir+img_lst[i])
    cv2_img_RGB=ksimg.img_to_array(cv2_img_RGB)
    #template_match_img=np.zeros((cv2_img_RGB.shape[0],cv2_img_RGB.shape[1]),np.uint8)
    #rows,cols=template_match_img.shape
    template_match_img = cv2_img_gray[:, int(cv2_img_RGB.shape[1]/2):cv2_img_RGB.shape[1]]
    #template_match_img_rgb = cv2_img_RGB[:, int(cv2_img_RGB.shape[1]/2):cv2_img_RGB.shape[1]]
    #print(template_match_img.shape)
    start_t=time.time()
    res = cv2.matchTemplate(template_match_img, template, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)#找出数组中的最大值最小值以及其对应的位置
    start_e=time.time()
   # print(start_e-start_t)
    # 如果是平方差匹配TM_SQDIFF或归一化平方差匹配TM_SQDIFF_NORMED，取最小值
    # if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    #     top_left = min_loc
    # else:
    #     top_left = max_loc
    #print(min_val)
    h,w=template.shape
    # print(template.shape)
    # print(top_left[0],top_left[1])
    top_left = (min_loc[0]+int(cv2_img_RGB.shape[1]/2),min_loc[1])
    bottom_right = (top_left[0] + w, top_left[1] + h)
    new_img=cv2_img_RGB[top_left[1]-2:top_left[1] + h+2,top_left[0]-2:top_left[0] + w+2]
    #new_img=cv2.cvtColor(new_img,cv2.COLOR_RGB2BGR)
    #cv2.imwrite('./Dataset/OK/{}'.format('1.jpg'),new_img)
    #cv2.rectangle(cv2_img_RGB, top_left, bottom_right, (0,255,0), 3)
    #cv2_img_RGB=cv2_img_RGB/255
#         plt.imshow(template_match_img) #imshow显示格式为image,为ndarrray时候，需要为整型
#         plt.show()
    reduce_part1=cv2_img_RGB[top_left[1]+100:top_left[1]+500, top_left[0]-100:top_left[0]+300,:]
    #print(reduce_part1)
#     plt.imshow(reduce_part1/255) #imshow显示格式为image,为ndarrray时候，需要为整型
#     plt.show()
    reduce_part2=cv2_img_RGB[top_left[1]:top_left[1]+400, top_left[0]-900:top_left[0]-300,:]
    #print(reduce_part1)
#     plt.imshow(reduce_part2/255) #imshow显示格式为image,为ndarrray时候，需要为整型
#     plt.show()
    print(cv2_img_RGB.shape)
    reduce_part3=cv2_img_RGB[top_left[1]:top_left[1]+400, top_left[0]-1400:top_left[0]-1000,:]
#     plt.imshow(reduce_part3/255) #imshow显示格式为image,为ndarrray时候，需要为整型
#     plt.show()
    reduce_part4=cv2_img_RGB[top_left[1]+50:top_left[1]+350, top_left[0]-1800:top_left[0]-1400,:]
#     plt.imshow(reduce_part4/255) #imshow显示格式为image,为ndarrray时候，需要为整型
#     plt.show()
    reduce_part5=cv2_img_RGB[top_left[1]+50:top_left[1]+400, top_left[0]-2100:top_left[0]-1800,:]
#     plt.imshow(reduce_part5/255) #imshow显示格式为image,为ndarrray时候，需要为整型
#     plt.show()
    reduce_part6=cv2_img_RGB[top_left[1]:top_left[1]+400, top_left[0]-2500:top_left[0]-2100,:]
#     plt.imshow(yolo.detect_image(reduce_part1)) #imshow显示格式为image,为ndarrray时候，需要为整型
#     plt.show()
    yolo.detect_image(reduce_part6)
    #yolo.close_session()