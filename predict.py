from yolo import YOLO
from PIL import Image
import  os
yolo = YOLO()

# while True:
#     img = input('Input image filename:')
#     try:
#         image = Image.open(img)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#         r_image = yolo.detect_image(image)
#         r_image.show()
# yolo.close_session()
img_path='result/nut_b'
yolo.detect_batchs_save(img_path)
yolo.close_session()
