import tensorflow as tf
import tensorflow.keras as keras
# K折交叉验证模块
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Conv3DTranspose
import os
import time
import numpy as np
from tensorflow.keras.utils import multi_gpu_model
from sklearn.preprocessing import StandardScaler

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# categorical_crossentropy binary_crossentropy mean_squared_error mean_absolute_error
loss_type = 'mean_squared_error'
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
GPU_COUNT = 5
batch_size = 32
n_epochs = 50  # (200)
LR = 0.005
valid_train_rate = 0.1
stride = [1, 1, 1]
padding = 'same'
IMAGE_HEIGHT = 72
IMAGE_WIDTH = 96
IMAGE_DEPTH = 64
volume_shape = [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH]
input_shape = [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1]
# input_shape2 = [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1]
dropout_flag = True
shuffle_flag = True
parallel_work = 2
# K.clear_session()
# tf.reset_default_graph()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

PROJ_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../"))
# filenames = DATA_DIR + '/output/fmri/tf_data/test_data_vol.tfrecords'
filenames = PROJ_DIR + '/output/fmri/tf_data/XXX.tfrecords'
LOG_DIR = PROJ_DIR + '/output/log'
loss_img_path = PROJ_DIR + "/output/images/loss_" + loss_type + "_" + str(n_epochs) + "_" + str(batch_size) + ".png"
acc_img_path = PROJ_DIR + "/output/images/acc_" + loss_type + "_" + str(n_epochs) + "_" + str(batch_size) + ".png"
output_model_file = PROJ_DIR + '/output/weights/autoencoder_mri.h5'
output_model_weight_file = PROJ_DIR + '/output/weights/autoencoder_weight_mri.h5'

scaler = StandardScaler()

callbacks = [
    keras.callbacks.TensorBoard(LOG_DIR),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                    save_best_only=True)  # 保存最好的模型，默认保存最近的
    # keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3),  # min_delta阈值：训练间的差别，
    # patience ：连续多少次发生min_delta的情况要停止
    # 比这个低的话，要停止
]
# print(tf.__version__)
# Create a description of the features.
feature_description = {
    'vol_raw': tf.io.FixedLenFeature([], tf.string),
    'sub-id': tf.io.FixedLenFeature([], tf.string)
}


def generate_array_from_file(filenames):
    X = []
    Y = []
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    parsed_dataset = dataset.map(_parse_function, num_parallel_calls=parallel_work)
    iterator = tf.compat.v1.data.make_one_shot_iterator(parsed_dataset)
    # cnt = 0

    try:
        while True:
            parsed_record = iterator.get_next()
            feature = parsed_record['vol_raw']
            # label = parsed_record['sub-id']
            # type:  'tensorflow.python.framework.ops.EagerTensor'
            vol_str = tf.compat.v1.decode_raw(feature, tf.float32)
            volume = tf.reshape(vol_str, volume_shape)
            a = volume.numpy()
            X.append(a)
            Y.append(a)
    except tf.errors.OutOfRangeError:
        return X
        pass


def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)


def Standardation(data, type):
    if type == 'valid':
        # 验证集 直接用transform
        return scaler.transform(
            data.astype(np.float32).reshape(-1, 1)).reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1])
    else:
        return scaler.fit_transform(
            data.astype(np.float32).reshape(-1, 1)).reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1])


def noramlization(data):
    minVals = data.min()
    maxVals = data.max()
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = normData.shape[0]
    normData = data - np.tile(minVals, (np.shape(data)))
    normData = normData / np.tile(ranges, (np.shape(data)))
    return normData, ranges, minVals


def autoencoder(input_img):
    # encoder
    x = Conv3D(data_format='channels_last', filters=32, kernel_size=(3, 3, 3), strides=stride, activation='relu',
               padding=padding)(input_img)
    x = MaxPooling3D(data_format='channels_last', pool_size=(2, 2, 2), strides=(3, 2, 2), padding=padding)(x)
    x = Conv3D(data_format='channels_last', filters=64, kernel_size=(3, 3, 3), strides=stride, activation='relu',
               padding=padding)(x)
    x = MaxPooling3D(data_format='channels_last', pool_size=(2, 2, 2), strides=(3, 3, 2), padding=padding)(x)
    x = Conv3D(data_format='channels_last', filters=16, kernel_size=(2, 2, 2), strides=stride, activation='relu',
               padding=padding)(x)
    x = MaxPooling3D(data_format='channels_last', pool_size=(2, 2, 2), strides=(1, 1, 2), padding=padding)(x)
    # print(x.shape)
    latent = Conv3D(data_format='channels_last', filters=1, kernel_size=(2, 2, 2), strides=stride, activation='relu',
                    padding=padding)(x)
    latent = Conv3DTranspose(data_format='channels_last', filters=1, kernel_size=(2, 2, 2), strides=stride,
                             activation='relu', padding=padding)(latent)

    # decoder
    x = UpSampling3D(data_format='channels_last', size=(1, 1, 2))(latent)
    x = Conv3DTranspose(data_format='channels_last', filters=16, kernel_size=(2, 2, 2), strides=stride,
                        activation='relu', padding=padding)(x)
    x = UpSampling3D(data_format='channels_last', size=(3, 3, 2))(x)
    x = Conv3DTranspose(data_format='channels_last', filters=64, kernel_size=(3, 3, 3), strides=stride,
                        activation='relu', padding=padding)(x)
    x = UpSampling3D(data_format='channels_last', size=(3, 2, 2))(x)
    decoded = Conv3DTranspose(data_format='channels_last', filters=32, kernel_size=(3, 3, 3), strides=stride,
                              activation='relu', padding=padding)(x)
    return decoded


def create_model():
    input_img = keras.Input(shape=input_shape)
    # (None, 72, 96, 64, 1)
    model = keras.Model(input_img, autoencoder(input_img))
    # auto_encoder = multi_gpu_model(auto_encoder, GPU_COUNT)
    # CE BCE MSE
    model.compile(optimizer='adam', loss=loss_type, metrics=['accuracy'])
    # auto_encoder.compile(optimizer='adam', metrics=['accuracy', 'ce'])
    model.summary()
    return model


def main():
    # data loading
    X = generate_array_from_file(filenames)
    # X, _, _ = noramlization(X)

    train_X, valid_X, train_ground, valid_ground = train_test_split(X,
                                                                    X,
                                                                    test_size=valid_train_rate,
                                                                    random_state=seed)
    # cvscores = []
    auto_encoder = create_model()
    # train_l =
    # train_r = np.array(X).reshape(-1,)
    # for train, valid in kfold.split(train_r):
        # train_x = train.reshape
        # valid_x = X[train.size:-1]

    train_x = np.array(train_X).reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1])
    train_x, _, _ = noramlization(train_x)

    valid_x = np.array(valid_X).reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1])
    valid_x, _, _ = noramlization(valid_x)

    # # 归一化  x = （x - u（均值）） / std(方差) 就得到符合均值是0,方差是1的
    #  正态分布
    x_train_scaled = Standardation(train_x, 'train')
    x_train_scaled, _, _ = noramlization(x_train_scaled)
    x_vaild_scaled = Standardation(valid_x, 'valid')
    x_vaild_scaled, _, _ = noramlization(x_vaild_scaled)

    autoencoder_train = auto_encoder.fit(x_train_scaled, train_x,
                                         # validation_data=0.1,
                                         batch_size=batch_size,
                                         epochs=n_epochs,
                                         shuffle=True,
                                         validation_data=(x_vaild_scaled, valid_x),
                                         verbose=1, callbacks=callbacks)

        # cvscores.append(scores[1] * 100)

    auto_encoder.save(output_model_file)
    auto_encoder.save_weights(output_model_weight_file)
    # Score trained model.

    # 中间层输出结果可视化
    # plt.figure()
    # 其中encoded_imgs[:,0]和encoded_imgs[:,1]代表提取出的两个特征
    # plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1])
    # plt.show()
    # scores = auto_encoder.evaluate(x_vaild_scaled, valid_X, batch_size=batch_size, verbose=2)
    # print('Test loss:', scores[0])
    # print('Test accuracy:', scores[1])

    loss = autoencoder_train.history['loss']
    val_loss = autoencoder_train.history['val_loss']
    accuracy = autoencoder_train.history['accuracy']
    val_accuracy = autoencoder_train.history['val_accuracy']
    # print("average train Loss:" + str(np.average(loss)) + ";average validation Loss:" + str(np.average(val_loss)))

    # epochs = range(300)

    ###################################################################
    plt.figure()
    plt.plot(np.arange(0, n_epochs), loss, linewidth='1', color='coral',
             marker='|', label='Training')
    plt.plot(np.arange(0, n_epochs), val_loss, color='green', linestyle='dashed', marker=',',
             markerfacecolor='blue', label='Validation')
    plt.title('Training and validation loss:epochs-' + str(n_epochs) + '-batch:' + str(batch_size))
    plt.legend(loc='upper left')
    plt.savefig(loss_img_path)
    # plt.show()
    # if loss_type == 'binary_crossentropy':
    plt.figure()
    plt.plot(np.arange(0, n_epochs), accuracy, linewidth='1', color='coral',
             marker='|', label='Training')
    plt.plot(np.arange(0, n_epochs), val_accuracy, color='green', linestyle='dashed', marker=',',
             markerfacecolor='blue', label='Validation')
    plt.title('Training and validation accuracy:epochs-' + str(n_epochs) + '-batch:' + str(batch_size))
    plt.legend(loc='upper left')
    plt.savefig(acc_img_path)
    # plt.show()


def make_print_to_file(path=LOG_DIR):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    import os
    # import config_file as cfg_file
    import sys
    import datetime

    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('day and time:' + '%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # print -> log
    #############################################################
    print(fileName.center(60, '*'))


if __name__ == '__main__':
    start = time.perf_counter()
    make_print_to_file(path='.')
    main()
    end = time.perf_counter()
    print("time-cost:", str((end - start) / 60))
