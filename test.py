import time
import  datetime
import keras.preprocessing.image as ksimg
import cv2
img=ksimg.load_img('./result/nut_b/0-1.jpg')
img=ksimg.img_to_array(img)
ksimg.save_img('1.jpg',img)