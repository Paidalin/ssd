import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.models import Model
from keras.regularizers import l2

from ssd_layers import Normalize
from ssd_layers import PriorBox

def SSD300(input_shape, num_classes=2):
    '''
        SSD300的框架，分类数目为2，包括背景。
    '''
    # 记录结构的字典
    net = {}
    l2_reg = 0.0005
    # Block1
    img_size = (input_shape[1], input_shape[0])
    net['input'] = Input(shape=input_shape)
    net['conv1_1'] = Conv2D(64, 3,
                            activation='relu',
                            padding='same', kernel_regularizer=l2(l2_reg),
                            name='conv1_1')(net['input'])
    net['conv1_2'] = Conv2D(64, 3,
                            activation='relu',
                            padding='same', kernel_regularizer=l2(l2_reg),
                            name='conv1_2')(net['conv1_1'])
    net['pool1'] = MaxPooling2D(2, 2, padding='same',
                                name='pool1')(net['conv1_2'])
    # Block2
    net['conv2_1'] = Conv2D(128, 3,
                            activation='relu',
                            padding='same', kernel_regularizer=l2(l2_reg),
                            name='conv2_1')(net['pool1'])
    net['conv2_2'] = Conv2D(128, 3,
                            activation='relu',
                            padding='same', kernel_regularizer=l2(l2_reg),
                            name='conv2_2')(net['conv2_1'])
    net['pool2'] = MaxPooling2D(2, 2, padding='same',
                                name='pool2')(net['conv2_2'])
    # Block3
    net['conv3_1'] = Conv2D(256, 3,
                            activation='relu',
                            padding='same', kernel_regularizer=l2(l2_reg),
                            name='conv3_1')(net['pool2'])
    net['conv3_2'] = Conv2D(256, 3,
                            activation='relu',
                            padding='same', kernel_regularizer=l2(l2_reg),
                            name='conv3_2')(net['conv3_1'])
    net['conv3_3'] = Conv2D(256, 3,
                            activation='relu',
                            padding='same', kernel_regularizer=l2(l2_reg),
                            name='conv3_3')(net['conv3_2'])
    net['pool3'] = MaxPooling2D(2, 2, padding='same',
                                name='pool3')(net['conv3_3'])
    # Block4
    net['conv4_1'] = Conv2D(512, 3,
                            activation='relu',
                            padding='same', kernel_regularizer=l2(l2_reg),
                            name='conv4_1')(net['pool3'])
    net['conv4_2'] = Conv2D(512, 3,
                            activation='relu',
                            padding='same', kernel_regularizer=l2(l2_reg),
                            name='conv4_2')(net['conv4_1'])
    net['conv4_3'] = Conv2D(512, 3,
                            activation='relu',
                            padding='same', kernel_regularizer=l2(l2_reg),
                            name='conv4_3')(net['conv4_2'])
    net['pool4'] = MaxPooling2D(2, 2, padding='same',
                                name='pool4')(net['conv4_3'])
    # Block5
    net['conv5_1'] = Conv2D(512, 3,
                            activation='relu',
                            padding='same', kernel_regularizer=l2(l2_reg),
                            name='conv5_1')(net['pool4'])
    net['conv5_2'] = Conv2D(512, 3,
                            activation='relu',
                            padding='same', kernel_regularizer=l2(l2_reg),
                            name='conv5_2')(net['conv5_1'])
    net['conv5_3'] = Conv2D(512, 3,
                            activation='relu',
                            padding='same', kernel_regularizer=l2(l2_reg),
                            name='conv5_3')(net['conv5_2'])
    net['pool5'] = MaxPooling2D(3, 1, padding='same',
                                name='pool5')(net['conv5_3'])
    # FC6
    net['fc6'] = Conv2D(1024, 3, dilation_rate=6,
                        activation='relu',
                        padding='same', kernel_regularizer=l2(l2_reg),
                        name='fc6')(net['pool5'])
    #FC7
    net['fc7'] = Conv2D(1024, 1,
                        activation='relu',
                        padding='same', kernel_regularizer=l2(l2_reg),
                        name='fc7')(net['fc6'])
    # Block6(pic.8)
    net['conv6_1'] = Conv2D(256, 1,
                            activation='relu',
                            padding='same', kernel_regularizer=l2(l2_reg),
                            name='conv6_1')(net['fc7'])
    net['conv6_1'] = ZeroPadding2D()(net['conv6_1'])
    net['conv6_2'] = Conv2D(512, 3, strides=2,
                            activation='relu',
                            padding='valid', kernel_regularizer=l2(l2_reg),
                            name='conv6_2')(net['conv6_1'])
    # Block7(pic.9)
    net['conv7_1'] = Conv2D(128, 1,
                            activation='relu',
                            padding='same', kernel_regularizer=l2(l2_reg),
                            name='conv7_1')(net['conv6_2'])
    net['conv7_2'] = ZeroPadding2D()(net['conv7_1'])
    net['conv7_2'] = Conv2D(256, 3, strides=2,
                            activation='relu',
                            padding='valid', kernel_regularizer=l2(l2_reg),
                            name='conv7_2')(net['conv7_2'])
    # Block8(pic.10)
    net['conv8_1'] = Conv2D(128, 1,
                            activation='relu',
                            padding='same', kernel_regularizer=l2(l2_reg),
                            name='conv8_1')(net['conv7_2'])
    # net['conv8_2'] = Conv2D(256, 3, strides=2,
    #                         activation='relu',
    #                         padding='same', kernel_regularizer=l2(l2_reg),
    #                         name='conv8_2')(net['conv8_1'])
    net['conv8_2'] = Conv2D(256, 3,
                            activation='relu',
                            padding='valid', kernel_regularizer=l2(l2_reg),
                            name='conv8_2')(net['conv8_1'])
    # Block9(pic.11)
    net['conv9_1'] = Conv2D(128, 1,
                            activation='relu',
                            padding='same', kernel_regularizer=l2(l2_reg),
                            name='conv9_1')(net['conv8_2'])
    net['conv9_2'] = Conv2D(256, 3,
                            activation='relu',
                            padding='valid', kernel_regularizer=l2(l2_reg),
                            name='conv9_2')(net['conv9_1'])
    # Last Pool(pic.11)
    # net['pool6'] = GlobalAveragePooling2D(name='pool6')(net['conv8_2'])
    
    # 从conv4_3预测
    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net['conv4_3'])
    num_priors = 3
    # 预测4个坐标
    net['conv4_3_norm_mbox_loc_lin'] = Conv2D(num_priors * 4, 3,
                                        padding='same', kernel_regularizer=l2(l2_reg),
                                        name='conv4_3_norm_mbox_loc_lin')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_loc_lin_flat'] = Flatten(name='conv4_3_norm_mbox_loc_lin_flat')(net['conv4_3_norm_mbox_loc_lin'])
    # 预测置信度
    net['conv4_3_norm_mbox_conf_lin'] = Conv2D(num_priors * num_classes, 3,
                                            padding='same', kernel_regularizer=l2(l2_reg),
                                            name='conv4_3_norm_mbox_conf_lin')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_conf_lin_flat'] = Flatten(name='conv4_3_norm_mbox_conf_lin_flat')(net['conv4_3_norm_mbox_conf_lin'])
    priorbox = PriorBox(img_size, 30.0, aspect_ratios=[2],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
    net['conv4_3_norm_mbox_priorbox'] = priorbox(net['conv4_3_norm'])
    # fc7
    num_priors = 6
    net['fc7_mbox_loc_lin'] = Conv2D(num_priors * 4, 3,
                                padding='same', kernel_regularizer=l2(l2_reg),
                                name='fc7_mbox_loc_lin')(net['fc7'])
    net['fc7_mbox_loc_lin_flat'] = Flatten(name='fc7_mbox_loc_lin_flat')(net['fc7_mbox_loc_lin'])
    net['fc7_mbox_conf_lin'] = Conv2D(num_priors * num_classes, 3,
                                padding='same', kernel_regularizer=l2(l2_reg),
                                name='fc7_mbox_conf_lin')(net['fc7'])
    net['fc7_mbox_conf_lin_flat'] = Flatten(name='fc7_mbox_conf_lin_flat')(net['fc7_mbox_conf_lin'])
    priorbox = PriorBox(img_size, 60.0, max_size=114.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='fc7_mbox_priorbox')
    net['fc7_mbox_priorbox'] = priorbox(net['fc7'])
    # conv6_2
    num_priors = 6
    net['conv6_2_mbox_loc_lin'] = Conv2D(num_priors * 4, 3,
                                    padding='same', kernel_regularizer=l2(l2_reg),
                                    name='conv6_2_mbox_loc_lin')(net['conv6_2'])
    net['conv6_2_mbox_loc_lin_flat'] = Flatten(name='conv6_2_mbox_loc_lin_flat')(net['conv6_2_mbox_loc_lin'])
    net['conv6_2_mbox_conf_lin'] = Conv2D(num_priors * num_classes, 3,
                                    padding='same', kernel_regularizer=l2(l2_reg),
                                    name='conv6_2_mbox_conf_lin')(net['conv6_2'])
    net['conv6_2_mbox_conf_lin_flat'] = Flatten(name='conv6_2_mbox_conf_lin_flat')(net['conv6_2_mbox_conf_lin'])
    priorbox = PriorBox(img_size, 114.0, max_size=168.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv6_2_mbox_priorbox')
    net['conv6_2_mbox_priorbox'] = priorbox(net['conv6_2'])
    # conv7_2
    num_priors = 6
    net['conv7_2_mbox_loc_lin'] = Conv2D(num_priors * 4, 3,
                                    padding='same', kernel_regularizer=l2(l2_reg),
                                    name='conv7_2_mbox_loc_lin')(net['conv7_2'])
    net['conv7_2_mbox_loc_lin_flat'] = Flatten(name='conv7_2_mbox_loc_lin_flat')(net['conv7_2_mbox_loc_lin'])
    net['conv7_2_mbox_conf_lin'] = Conv2D(num_priors * num_classes, 3,
                                    padding='same', kernel_regularizer=l2(l2_reg),
                                    name='conv7_2_mbox_conf_lin')(net['conv7_2'])
    net['conv7_2_mbox_conf_lin_flat'] = Flatten(name='conv7_2_mbox_conf_lin_flat')(net['conv7_2_mbox_conf_lin'])
    priorbox = PriorBox(img_size, 168.0, max_size=222.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv7_2_mbox_priorbox')
    net['conv7_2_mbox_priorbox'] = priorbox(net['conv7_2'])
    # conv8_2
    num_priors = 6
    net['conv8_2_mbox_loc_lin'] = Conv2D(num_priors * 4, 3,
                                    padding='same', kernel_regularizer=l2(l2_reg),
                                    name='conv8_2_mbox_loc_lin')(net['conv8_2'])
    net['conv8_2_mbox_loc_lin_flat'] = Flatten(name='conv8_2_mbox_loc_lin_flat')(net['conv8_2_mbox_loc_lin'])
    net['conv8_2_mbox_conf_lin'] = Conv2D(num_priors * num_classes, 3,
                                    padding='same', kernel_regularizer=l2(l2_reg),
                                    name='conv8_2_mbox_conf_lin')(net['conv8_2'])
    net['conv8_2_mbox_conf_lin_flat'] = Flatten(name='conv8_2_mbox_conf_lin_flat')(net['conv8_2_mbox_conf_lin'])
    priorbox = PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv8_2_mbox_priorbox')
    net['conv8_2_mbox_priorbox'] = priorbox(net['conv8_2'])
    # # pool6
    # num_priors = 6
    # net['pool6_mbox_loc_lin_flat'] = Dense(num_priors * 4, name='pool6_mbox_loc_lin_flat')(net['pool6'])
    # net['pool6_mbox_conf_lin_flat'] = Dense(num_priors * num_classes, name='pool6_mbox_conf_lin_flat')(net['pool6'])
    # priorbox = PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3],
    #                     variances=[0.1, 0.1, 0.2, 0.2],
    #                     name='pool6_mbox_priorbox')
    # net['pool6_reshaped'] = Reshape((1, 1, 256),
    #                                 name='pool6_reshaped')(net['pool6'])
    # net['pool6_mbox_priorbox'] = priorbox(net['pool6_reshaped'])
    # conv9_2
    num_priors = 6
    net['conv9_2_mbox_loc_lin'] = Conv2D(num_priors * 4, 3,
                                    padding='same', kernel_regularizer=l2(l2_reg),
                                    name='conv9_2_mbox_loc_lin')(net['conv9_2'])
    net['conv9_2_mbox_loc_lin_flat'] = Flatten(name='conv9_2_mbox_loc_lin_flat')(net['conv9_2_mbox_loc_lin'])
    net['conv9_2_mbox_conf_lin'] = Conv2D(num_priors * num_classes, 3,
                                    padding='same', kernel_regularizer=l2(l2_reg),
                                    name='conv9_2_mbox_conf_lin')(net['conv9_2'])
    net['conv9_2_mbox_conf_lin_flat'] = Flatten(name='conv9_2_mbox_conf_lin_flat')(net['conv9_2_mbox_conf_lin'])
    priorbox = PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv9_2_mbox_priorbox')
    net['conv9_2_mbox_priorbox'] = priorbox(net['conv9_2'])

    net['mbox_loc_lin'] = concatenate([net['conv4_3_norm_mbox_loc_lin_flat'],
                                    net['fc7_mbox_loc_lin_flat'],
                                    net['conv6_2_mbox_loc_lin_flat'],
                                    net['conv7_2_mbox_loc_lin_flat'],
                                    net['conv8_2_mbox_loc_lin_flat'],
                                    net['conv9_2_mbox_loc_lin_flat']],
                                    axis=1, name='mbox_loc_lin')
    net['mbox_conf_lin'] = concatenate([net['conv4_3_norm_mbox_conf_lin_flat'],
                                    net['fc7_mbox_conf_lin_flat'],
                                    net['conv6_2_mbox_conf_lin_flat'],
                                    net['conv7_2_mbox_conf_lin_flat'],
                                    net['conv8_2_mbox_conf_lin_flat'],
                                    net['conv9_2_mbox_conf_lin_flat']],
                                    axis=1, name='mbox_conf_lin')
    net['mbox_priorbox'] = concatenate([net['conv4_3_norm_mbox_priorbox'],
                                        net['fc7_mbox_priorbox'],
                                        net['conv6_2_mbox_priorbox'],
                                        net['conv7_2_mbox_priorbox'],
                                        net['conv8_2_mbox_priorbox'],
                                        net['conv9_2_mbox_priorbox']],
                                        axis=1, name='mbox_priorbox')
    if hasattr(net['mbox_loc_lin'], '_keras_shape'):
        num_boxes = net['mbox_loc_lin']._keras_shape[-1] // 4
    elif hasattr(net['mbox_loc_lin'], 'int_shape'):
        num_boxes = K.int_shape(net['mbox_loc_lin'])[-1] // 4
    net['mbox_loc_lin'] = Reshape((num_boxes, 4),
                              name='mbox_loc_lin_final')(net['mbox_loc_lin'])
    net['mbox_conf_lin'] = Reshape((num_boxes, num_classes),
                               name='mbox_conf_lin_logits')(net['mbox_conf_lin'])
    net['mbox_conf_lin'] = Activation('softmax',
                                  name='mbox_conf_lin_final')(net['mbox_conf_lin'])
    net['predictions'] = concatenate([net['mbox_loc_lin'],
                                    net['mbox_conf_lin'],
                                    net['mbox_priorbox']],
                                    axis=2, name='predictions')
    model = Model(net['input'], net['predictions'])
    return model
