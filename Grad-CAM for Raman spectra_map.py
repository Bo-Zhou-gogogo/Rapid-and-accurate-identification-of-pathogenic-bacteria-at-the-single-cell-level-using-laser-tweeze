# GPU按需分配,解决
import tensorflow as tf
import keras

from tensorflow.python.keras.models import load_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
from keras import backend as K
import os
import numpy as np
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


x_test = np.load('./data/test_data_average.npy')
print('test dataset shape', x_test.shape)
y_test = np.load('./data/test_label_average.npy')
print('test dataset shape', y_test.shape)

import pandas as pd

data_first = x_test[0,:].T
data_first = data_first[:,None]
heatmap_first = x_test[0].T
heatmap_first = heatmap_first[:,None]

for label in range(len(x_test)):
    w = 0
    nb_features = 861
    print(x_test.shape)
    data_test = x_test[label:label + 1]
    X_test_r = np.zeros((len(data_test), nb_features, 1))
    X_test_r[:, :, 0] = data_test[:, :nb_features]

    # Grad-CAM包含三个变量和两个运算：
    # 变量：1.卷积层输出特征con_layer.output; 2.卷积层输出特征的每个通道con_layer.output[i]；3.特定输出类别model.output[i]
    # 运算：1.求类别对于卷积层特征梯度K.gradients(y,x)-----K.gradients(model.output[i], con_layer.output
    #     对上述梯度平均化：K.mean(gradients)------------K.mean(gradients, axis=(0,1,2))
    #
    # 2.乘梯度:卷积特征图乘以平均化后的梯度：con_layer.output[i]*gradients
    # 热力图：np.mean()

    # 预测
    session = tf.Session(graph=tf.Graph())
    global graph1
    graph1 = tf.get_default_graph()
    with session.as_default():  # 强制在一个线程
        with session.graph.as_default():
            K.set_session(session)
            model = load_model('./2_model_add.hdf5')

            model.summary()
            y_pre_i = model.predict(X_test_r)  # one-hot矩阵
            print('the {} spcies, real label is{}, predicted label is{}'.format(label, y_test[label], np.argmax(y_pre_i)))

            # 类别输出
            class_output = model.output[:, int(y_test[label])]
            # 卷积层输出特征图

            convolution_output = model.get_layer('batch_normalization_17').output
    # convolution_output = model.layers['activation_17']

    # 类别相对于 卷积层特征图的梯度
    with session.as_default():
        with session.graph.as_default():
            grads = K.gradients(class_output, convolution_output)[0]
            gradient_function = K.function([model.input], [convolution_output, grads])

            output, grads_val = gradient_function([X_test_r])

            output, grads_val = output[0], grads_val[0]
    # 取平均

    weights = np.mean(grads_val, axis=(0))
    for i in range(512):
        output[:, i] *= weights[i]

    # 得到的特征图的逐通道平均值即为类激活的热力图
    heatmap = np.mean(output, axis=-1)
    # print(heatmap)


    heatmap = np.maximum(heatmap, 0)  # X和Y逐位进行比较,选择最大值。保留大于零的数值
    heatmap /= np.max(heatmap)  # npmax 求序列的最值，axis默认为axis=0即列向。heatmap每一列除以最大值，完成归一化
    # plt.matshow(heatmap)
    # plt.show()

    heat_bp = heatmap


    # print(data_test.shape[1],data_test.shape[0])
    heatmap = cv2.resize(heatmap, (data_test.shape[0], data_test.shape[1]))  # 插值扩大形状，cv2.resize(输入尺寸，输出尺寸)
    # heatmap = np.uint8(255 * heatmap)
    # 将热力图应用于原始图像
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # 这里的 0.4 是热力图强度因子

    # cv2.imwrite('elephant_cam_last.jpg', heatmap)
    import pandas as pd

    data_test = data_test.T

    data_first = np.concatenate([data_first, data_test], axis=1)

    heatmap_first = np.concatenate([heatmap_first, heatmap], axis=1)

spectra_data = pd.DataFrame(data=data_first[:, 1:])
spectra_data.to_csv('./data/spectra_data_test.csv')

heatmap_all = pd.DataFrame(data=heatmap_first[:, 1:])
heatmap_all.to_csv('./data/heatmap_test.csv')
