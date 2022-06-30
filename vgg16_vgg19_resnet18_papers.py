import os, shutil
import tensorflow as tf
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.preprocessing import image
import cv2
import numpy as np
from keras.layers import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Dropout
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,Input,LeakyReLU,ZeroPadding2D,BatchNormalization,add,AveragePooling2D
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from utils.utils import get_random_data,get_random_data_with_Mosaic,rand,WarmUpCosineDecayScheduler
import keras.backend as K
from keras.models import load_model
import random
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import  confusion_matrix,recall_score,classification_report
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from keras.layers import Lambda
from keras.callbacks import (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
                             TensorBoard)
from utils.utils import get_random_data,get_random_data_with_Mosaic,rand,WarmUpCosineDecayScheduler
## 准备数据


def data_div_and_prepare():
    original_ok_dir = './datasets/ok/'
    original_fail_dir = './datasets/ng/'
    base_dir = './datasets'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')
    train_ok_dir = os.path.join(train_dir, 'OK')
    train_fail_dir = os.path.join(train_dir, 'ng')
    validation_ok_dir = os.path.join(validation_dir, 'ok')
    validation_fail_dir = os.path.join(validation_dir, 'ng')
    test_ok_dir = os.path.join(test_dir, 'ok')
    test_fail_dir = os.path.join(test_dir, 'ng')
    ok_filename_lst = []
    for filename in os.listdir(original_ok_dir):
        ok_filename_lst.append(filename)
    fail_filename_lst = []
    for filename in os.listdir(original_fail_dir):
        fail_filename_lst.append(filename)
    random.shuffle(ok_filename_lst)
    random.shuffle(fail_filename_lst)
    ok_train_len = int(len(ok_filename_lst) * 0.7)
    fail_train_len = int(len(fail_filename_lst) * 0.7)
    ok_val_len = int(len(ok_filename_lst) * 0.2)
    fail_val_len = int(len(fail_filename_lst) * 0.2)

## VGG19
def VGG19_body():
    w = 50
    h = 50
    base_model = VGG19(include_top=False, input_shape=(w, h, 3))
    x = base_model.output
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    predictions = Dense(2, activation='softmax')(x)  # new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return  model
# def add_new_last_layer(base_model, nb_classes, drop_rate=0.):
#     """Add last layer to the convnet
#     Args:
#         base_model: keras model excluding top
#         nb_classes: # of classes
#     Returns:
#         new keras model with last layer
#     """
# #     inputs = Input(shape=(50,50,1))
# #     basemodelinput = layers.Conv2D(3, 1, padding = 'same')(inputs)
#     x = base_model.output
#     x = Dropout(0.5)(x)
#     x = Flatten()(x)
#     predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
#     model = Model(input=base_model.input, output=predictions)
#     return model
# w=50
# h=50
# base_model = VGG19(include_top=False,input_shape=(w, h, 3))
# model = add_new_last_layer(base_model, 2)
# model.summary()

## VGG16
def VGG19_body():
    w = 50
    h = 50
    base_model = VGG16(include_top=False, input_shape=(w, h, 3))
    x = base_model.output
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    predictions = Dense(2, activation='softmax')(x)  # new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return  model

## VGG16
# def add_new_last_layer(base_model, nb_classes, drop_rate=0.):
#     """Add last layer to the convnet
#     Args:
#         base_model: keras model excluding top
#         nb_classes: # of classes
#     Returns:
#         new keras model with last layer
#     """
# #     inputs = Input(shape=(50,50,1))
# #     basemodelinput = layers.Conv2D(3, 1, padding = 'same')(inputs)
#     x = base_model.output
#     x = Dropout(0.5)(x)
#     x = Flatten()(x)
#     predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
#     model = Model(input=base_model.input, output=predictions)
#     return model
# w=50
# h=50
# base_model = VGG16(include_top=False,input_shape=(w, h, 3))
# model_vgg16 = add_new_last_layer(base_model, 2)
# model_vgg16.summary()
def conv_block(inputs,
               neuron_num,
               kernel_size,
               use_bias,
               padding='same',
               strides=(1, 1),
               with_conv_short_cut=False):
    conv1 = Conv2D(
        neuron_num,
        kernel_size=kernel_size,
        activation='relu',
        strides=strides,
        use_bias=use_bias,
        padding=padding
    )(inputs)
    conv1 = BatchNormalization(axis=1)(conv1)

    conv2 = Conv2D(
        neuron_num,
        kernel_size=kernel_size,
        activation='relu',
        use_bias=use_bias,
        padding=padding)(conv1)
    conv2 = BatchNormalization(axis=1)(conv2)

    if with_conv_short_cut:
        inputs = Conv2D(
            neuron_num,
            kernel_size=kernel_size,
            strides=strides,
            use_bias=use_bias,
            padding=padding
        )(inputs)
        return add([inputs, conv2])

    else:
        return add([inputs, conv2])

## ResNET18
def resnet18_body():
    w=112
    h=112

    inputs = Input(shape= [112, 112, 1])
    x = ZeroPadding2D((3, 3))(inputs)

    # Define the converlutional block 1
    x = Conv2D(64, kernel_size= (7, 7), strides= (2, 2), padding= 'valid')(x)
    x = BatchNormalization(axis= 1)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D()(x)
    x= layers.MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)


    # Define the converlutional block 2
    x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)
    x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)
    # Define the converlutional block 3
    x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
    x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True)
    # Define the converlutional block 4
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
    x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
    # Define the converltional block 5
    x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
    x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True)
    x = AveragePooling2D(pool_size=(4, 4))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)

    model_resnet18 = Model(inputs= inputs, outputs= x)
    return model_resnet18


## ResNET34
#
#
# w=112
# h=112
# def conv_block(inputs,
#         neuron_num,
#         kernel_size,
#         use_bias,
#         padding= 'same',
#         strides= (1, 1),
#         with_conv_short_cut = False):
#     conv1 = Conv2D(
#         neuron_num,
#         kernel_size = kernel_size,
#         activation= 'relu',
#         strides= strides,
#         use_bias= use_bias,
#         padding= padding
#     )(inputs)
#     conv1 = BatchNormalization(axis = 1)(conv1)
#
#     conv2 = Conv2D(
#         neuron_num,
#         kernel_size= kernel_size,
#         activation= 'relu',
#         use_bias= use_bias,
#         padding= padding)(conv1)
#     conv2 = BatchNormalization(axis = 1)(conv2)
#
#     if with_conv_short_cut:
#         inputs = Conv2D(
#             neuron_num,
#             kernel_size= kernel_size,
#             strides= strides,
#             use_bias= use_bias,
#             padding= padding
#             )(inputs)
#         return add([inputs, conv2])
#
#     else:
#         return add([inputs, conv2])
#
# inputs = Input(shape= [112, 112, 1])
# x = ZeroPadding2D((3, 3))(inputs)
#
# # Define the converlutional block 1
# x = Conv2D(64, kernel_size= (7, 7), strides= (2, 2), padding= 'valid')(x)
# x = BatchNormalization(axis= 1)(x)
#
# # Define the converlutional block 2
# x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)
# x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)
# x = conv_block(x, neuron_num= 64, kernel_size= (3, 3), use_bias= True)
#
# # Define the converlutional block 3
# x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
# x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True)
# x = conv_block(x, neuron_num= 128, kernel_size= (3, 3), use_bias= True)
#
# # Define the converlutional block 4
# x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
# x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
# x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
# x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
# x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
# x = conv_block(x, neuron_num= 256, kernel_size= (3, 3), use_bias= True)
#
# # Define the converltional block 5
# x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True, strides= (2, 2), with_conv_short_cut= True)
# x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True)
# x = conv_block(x, neuron_num= 512, kernel_size= (3, 3), use_bias= True)
# x = AveragePooling2D(pool_size=(4, 4))(x)
# x = Flatten()(x)
# x = Dropout(0.5)(x)
# x = Dense(2, activation='softmax')(x)
#
# model = Model(inputs= inputs, outputs= x)
# model.summary()
#


## 论文模型

def kd_Studentnet_body():
    w=50
    h=50
    inputs = Input(shape=(50,50,1))
    x = layers.Conv2D(32, 1, padding = 'same')(inputs)
    x = layers.Conv2D(32, 3, padding = 'same')(inputs)
    x = layers.Conv2D(64, 1, padding = 'same')(inputs)
    x = layers.Conv2D(64, 3, padding = 'same')(inputs)
    x=layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    block_1_output = layers.MaxPooling2D(2)(x)
    #resblock 1
    block_1_output = layers.Conv2D(128, 1, padding = 'same')(block_1_output)
    x = layers.Conv2D(128, 3, padding = 'same')(block_1_output)
    x= layers.Conv2D(128, 3, padding = 'same')(x)
    x = layers.add([x,block_1_output])
    x=layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    block_2_output = layers.MaxPooling2D(2)(x)
    #resblock 2
    block_2_output = layers.Conv2D(256, 1, padding = 'same')(block_2_output)
    x = layers.Conv2D(256, 3, padding = 'same')(block_2_output)
    x= layers.Conv2D(256, 3, padding = 'same')(x)
    x = layers.add([x,block_2_output])
    x=layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    block_3_output = layers.MaxPooling2D(2)(x)
    #resblock 3
    block_3_output = layers.Conv2D(512, 1, padding = 'same')(block_3_output)
    x = layers.Conv2D(512, 3, padding = 'same')(block_3_output)
    x= layers.Conv2D(512, 3, padding = 'same')(x)
    x = layers.add([x,block_3_output])
    x=layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    block_4_output = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128)(block_4_output)
    x=layers.BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = layers.Dense(2,activation="softmax")(x)
    model = Model(inputs,x)
    return model




##VGGdatapreprocess
def VGGdatapreprocess(batch_size,train_dir,validation_dir):
    # batch_size=128
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     zoom_range=0.2,
        horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.b(
        train_dir,
    #     color_mode = 'grayscale',
        target_size=(50, 50),
        batch_size=batch_size,
        classes=['ok', 'ng'],
        class_mode='categorical'
        )
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
    #     color_mode = 'grayscale',
        target_size=(50, 50),
        batch_size=batch_size,
        classes=['ok', 'ng'],
        class_mode='categorical'
        )




##  resnet and studentnet preprocess
def resnet_studentnet_preprocess(batch_size,train_dir,validation_dir):
    # batch_size=64
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.2,
    #     zoom_range=0.2,
        horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        color_mode = 'grayscale',
        target_size=(112, 112),
        batch_size=batch_size,
        classes=['ok', 'ng'],
        class_mode='categorical'
        )
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        color_mode = 'grayscale',
        target_size=(112, 112),
        batch_size=batch_size,
        classes=['ok', 'ng'],
        class_mode='categorical'
        )


# train_len=ok_train_len+fail_train_len
# print(train_len)
# val_len=ok_val_len+fail_val_len
# print(val_len)
# epoch=100
# log_dir='./log/'
# learning_rate_base = 1e-3
#
# warmup_epoch = int(epoch*0.2)
# # 总共的步长
# total_steps = int(epoch * train_len / batch_size)
# # 预热步长
# warmup_steps = int(warmup_epoch * train_len / batch_size)
# 学习率
# reduce_lr = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base,
#                                         total_steps=total_steps,
#                                         warmup_learning_rate=1e-6,
# #                                         hold_base_rate_steps=warmup_steps//4,
#                                         warmup_steps=warmup_steps,
#                                         min_learn_rate=1e-7
#                                         )
 #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
# logging = TensorBoard(log_dir=log_dir)
# checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
#     monitor='val_loss', save_weights_only=True, save_best_only=True, period=20)
# reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=5, verbose=1,min_lr=1e-8)
# early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=10, verbose=1)
#
#
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=train_len/batch_size,
#     epochs=epoch,
#     validation_data=validation_generator,
#     validation_steps=val_len/batch_size,
#     callbacks=[logging, checkpoint, reduce_lr,early_stopping],
#     verbose=1)
#     # class_weight=[1, 100]
#
#
# yaml_string = model.to_yaml()
# with open("logs/model_vgg19.yaml", "w") as f:
#     f.write(yaml_string)
#
# pre_filename_lst_ok = []
# for filename in os.listdir('./datasets/test/ok'):
#     pre_filename_lst_ok.append(filename)
# pre_filename_lst_ng = []
# for filename in os.listdir('./datasets/test/ng'):
#     pre_filename_lst_ng.append(filename)
# import cv2
# def calpercent(path):
#     img = cv2.imread(path)
#     img=cv2.resize(img,(50,50))
#     img_tensor = image.img_to_array(img)
#     img_tensor = np.expand_dims(img_tensor, axis=0)
#     img_tensor /= 255.
#     prediction = model.predict(img_tensor)
#     label = prediction.argmax()
#     if label:
#         res = 0
#     else:
#         res = 1
#     return res
# import time
# count=0
# test_sample_oklst=[]
# test_predict_sample_oklst=[]
# start = time.time()
# for i in range(len(pre_filename_lst_ok)):
#     path=('./datasets/test/ok/'+pre_filename_lst_ok[i])
#     count=count+calpercent(path)
#     test_sample_oklst.append(1)
#     test_predict_sample_oklst.append(calpercent(path))
# end = time.time()
# print( '%.2f'% (end-start))
# #     print(predict(path))
# # print(str(round(count/len(pre_filename_lst2),3)))
# count=0
# test_sample_nglst=[]
# test_predict_sample_nglst=[]
# for i in range(len(pre_filename_lst_ng)):
#     path=('./datasets/test/ng/'+pre_filename_lst_ng[i])
#     count=count+calpercent(path)
#     test_sample_nglst.append(0)
#     test_predict_sample_nglst.append(calpercent(path))
# # print(str(1-round(count/len(pre_filename_lst_ng),3)))
# test_predict_lst=test_predict_sample_oklst+test_predict_sample_nglst
# test_lst=test_sample_oklst+test_sample_nglst
#
#
# cnf_matrix=confusion_matrix(test_lst,test_predict_lst)
# # print("Recall metric in the testing dataset:",cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
# class_names=[0,1]
# plt.figure()
# sns.heatmap(cnf_matrix, annot=True,fmt='g')
# plt.title('confusion_matrix')
# plt.show()


def get_kd_loss(args, temperature,
                alpha):
    student_logits = args[0:1][0]
    teacher_logits = args[1:2][0]
    true_labels = args[2:3]
    #     print(student_logits)
    #     print(args[0:1])
    #     teacher_logits_label=K.max(teacher_logits)
    #     student_logits_label=K.max(student_logits)
    teacher_probs = tf.divide(teacher_logits, temperature)
    kd_loss = tf.keras.losses.categorical_crossentropy(
        teacher_probs, tf.divide(student_logits, temperature),
        from_logits=True)

    ce_loss = tf.keras.losses.categorical_crossentropy(
        true_labels, student_logits, from_logits=True)

    total_loss = (alpha * kd_loss) + (1 - alpha) * ce_loss
    return total_loss

# temperature=5
# alpha=0.7
#
#
# y_true=Input(shape=(2,))
# teacher_logits=Input(shape=(2,))
# loss_input = [model.output, teacher_logits,y_true]
# print(loss_input)
# model_loss = Lambda(get_kd_loss, output_shape=(1,), name='model_loss',
#     arguments={'temperature': temperature, 'alpha': alpha})(loss_input)
#
# model_kdstudent = Model([model.input,y_true,teacher_logits], model_loss)
#
#
# model_kdstudent.compile(loss={'model_loss': lambda y_true, y_pred: y_pred}, optimizer=optimizers.Adam(lr=1e-4,beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0))
#
#
def get_data(dir_path):
    img = cv2.imread(dir_path, 0)
    img = cv2.resize(img, (50, 50))
    img = img * 1. / 255
    return img
#
def data_generator(dir_path, batch_size, input_shape):
    '''data generator for fit_generator'''
    i=0
    file_lst=os.listdir(dir_path)
    n=len(file_lst)
    while True:
        if i%n==0:
            np.random.shuffle(dir_path)
        image_data = []
        for b in range(batch_size):
            image= get_data(file_lst[i], input_shape)
            i = (i+1) % n
            image_data.append(image)
        image_data = np.array(image_data)
        y_true =1
        yield [image_data, *y_true], np.zeros(batch_size)
#
#
#  model.fit_generator(data_generator(train_dir, batch_size, input_shape),
#               steps_per_epoch=train_len/batch_size,
#     epochs=epoch,
#     validation_data=data_generator(val_dir, batch_size, input_shape),
#     validation_steps=val_len/batch_size,
#     callbacks=[logging, checkpoint, reduce_lr,early_stopping],
#     verbose=1)


if __name__ == "__main__":
    data_div_and_prepare()
    model=kd_Studentnet_body()
    temperature=5
    alpha=0.7
    y_true=Input(shape=(2,))
    teacher_logits=Input(shape=(2,))
    loss_input = [model.output, teacher_logits,y_true]
    print(loss_input)
    model_loss = Lambda(get_kd_loss, output_shape=(1,), name='model_loss',
        arguments={'temperature': temperature, 'alpha': alpha})(loss_input)
    model_kdstudent = Model([model.input,y_true,teacher_logits], model_loss)
    model_kdstudent.compile(loss={'model_loss': lambda y_true, y_pred: y_pred}, optimizer=optimizers.Adam(lr=1e-4,beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0))

