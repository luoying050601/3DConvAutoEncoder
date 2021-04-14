# napier
# fermat
import os
import time
import json
import numpy as np
import scipy.io as io
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import losses
import tensorflow.keras.backend as K
from matplotlib import pyplot as plt
import tensorflow.keras.layers as layers
from sklearn.preprocessing import StandardScaler, Normalizer
# K折交叉验证模块
from sklearn.model_selection import train_test_split
# from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Conv3DTranspose

# from tensorflow.keras.kerautils import multi_gpu_model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
#  binary_crossentropy mean_squared_error cosine_similarity
loss_type = 'mean_squared_error'
# kfold = StratifiedKFold(n_splits=4, random_state=seed)

batch_size = 32
n_splits = 10
n_epochs = 100  # (200)
LR = 0.005
valid_train_rate = 0.1
stride = [1, 1, 1]
padding = 'same'
IMAGE_HEIGHT = 72
IMAGE_WIDTH = 96
IMAGE_DEPTH = 64
volume_shape = [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH]
input_shape = [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1]
dropout_flag = True
shuffle_flag = True
parallel_work = 2
# K.clear_session()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
GPU_COUNT = 4

participants = {'18', '22', '23', '24', '26', '28', '30', '31', '35', '36', '37',
                '39', '41', '42', '43', '44', '45', '47', '48', '49', '50', '51', '53'}
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

PROJ_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../"))
LOG_DIR = PROJ_DIR + '/output/log'
loss_img_path = PROJ_DIR + "/output/images/loss_ab_" + loss_type + "_" + str(n_epochs) + "_" + str(batch_size) + ".png"
acc_img_path = PROJ_DIR + "/output/images/acc_ab_" + loss_type + "_" + str(n_epochs) + "_" + str(batch_size) + ".png"
output_model_file = PROJ_DIR + '/output/weights/autoencoder_ab_kf.h5'
# annotation_file = PROJ_DIR + '/output/annotation/annotations.tsv'
output_model_weight_file = PROJ_DIR + '/output/weights/autoencoder_weight_ab_kf.h5'
SCORE_JSON = PROJ_DIR + '/output/weights/score.json'
loss_dic = {}
acc_dic = {}
vloss_dic = {}
vacc_dic = {}
count = 0


# Custom loss layer
def l1_loss(x, v, c=count):
    x = K.flatten(x)
    v = K.flatten(v[c])
    count = 1 + c
    # print(x.shape)
    # print(v.shape)
    return K.mean(losses.mean_absolute_error(x, v))


class CustomVariationalLayer(layers.Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def __call__(self, inputs):
        # x = inputs[0]
        latent = Conv3D(data_format='channels_last', filters=1, kernel_size=(2, 2, 2), strides=stride,
                        activation='relu',
                        padding=padding, name='latent')(inputs)
        v = load_text_embedding_list()
        v, _, _ = normalization(v)

        loss = l1_loss(inputs, v)
        self.add_loss(loss, inputs=inputs)
        return latent


def autoencoder(input_img):
    # encoder
    x = Conv3D(data_format='channels_last', filters=32, kernel_size=(3, 3, 3), strides=stride, activation='relu',
               padding=padding, name='encode')(input_img)
    x = MaxPooling3D(data_format='channels_last', pool_size=(2, 2, 2), strides=(3, 2, 2), padding=padding)(x)
    x = Conv3D(data_format='channels_last', filters=64, kernel_size=(3, 3, 3), strides=stride, activation='relu',
               padding=padding)(x)
    x = MaxPooling3D(data_format='channels_last', pool_size=(2, 2, 2), strides=(3, 3, 2), padding=padding)(x)
    x = Conv3D(data_format='channels_last', filters=16, kernel_size=(2, 2, 2), strides=stride, activation='relu',
               padding=padding)(x)
    x = MaxPooling3D(data_format='channels_last', pool_size=(2, 2, 2), strides=(1, 1, 2), padding=padding)(x)

    # latent = Conv3D(data_format='channels_last', filters=1, kernel_size=(2, 2, 2), strides=stride, activation='relu',
    #                 padding=padding, name='latent')(x)
    latent = CustomVariationalLayer()(x)
    return latent


def autodecoder(latent):
    # decoder
    latent = Conv3DTranspose(data_format='channels_last', filters=1, kernel_size=(2, 2, 2), strides=stride,
                             activation='relu', padding=padding)(latent)
    x = UpSampling3D(data_format='channels_last', size=(1, 1, 2))(latent)
    x = Conv3DTranspose(data_format='channels_last', filters=16, kernel_size=(2, 2, 2), strides=stride,
                        activation='relu', padding=padding)(x)
    x = UpSampling3D(data_format='channels_last', size=(3, 3, 2))(x)
    x = Conv3DTranspose(data_format='channels_last', filters=64, kernel_size=(3, 3, 3), strides=stride,
                        activation='relu', padding=padding)(x)
    x = UpSampling3D(data_format='channels_last', size=(3, 2, 2))(x)
    decoded = Conv3DTranspose(data_format='channels_last', filters=32, kernel_size=(3, 3, 3), strides=stride,
                              activation='relu', padding=padding, name='decode')(x)
    # softmax trying(seems not good)
    # relu good
    # sigmoid (seems not good)
    return decoded


def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    stamp = ("".join(time_stamp.split()[0].split("-")) + "".join(time_stamp.split()[1].split(":"))).replace('.', '')
    return stamp


callbacks = [
    keras.callbacks.TensorBoard(LOG_DIR),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                    save_best_only=True)  # 保存最好的模型，默认保存最近的
    # keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
]
# Create a description of the features.
feature_description = {
    'vol_raw': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
    'sub-id': tf.io.FixedLenFeature([], tf.string)
}


def load_text_embedding_list():
    data = io.loadmat('text_embedding.mat')
    return data['result']


#
def write_score(a, b, c, d, time):
    timestamp = get_time_stamp()
    result = {}
    with open(SCORE_JSON, 'r', encoding='utf-8') as f:
        feeds = json.load(f)

    with open(SCORE_JSON, 'w', encoding='utf-8') as fp:
        result[timestamp] = {
            'epoch': n_epochs,
            'batch_size': batch_size,
            'loss_type': loss_type,
            'time_cost': time,
            'score': {
                'loss': a,
                'accurary': b,
                'val_loss': c,
                'val_accurary': d
            }
        }
        feeds.update(result)
        json.dump(feeds, fp, ensure_ascii=False)


# 解析tfrecord 文件
def generate_array_from_file(filenames):
    X = []
    Y = []
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    parsed_dataset = dataset.map(_parse_function, num_parallel_calls=parallel_work)
    iterator = tf.compat.v1.data.make_one_shot_iterator(parsed_dataset)
    cnt = 0

    try:
        while True:
            cnt = cnt + 1
            parsed_record = iterator.get_next()
            # type:  'tensorflow.python.framework.ops.EagerTensor'
            feature = tf.compat.v1.decode_raw(parsed_record['vol_raw'], tf.float32)
            # feature = parsed_record['vol_raw']
            label = parsed_record['label']
            volume = tf.reshape(feature, volume_shape)
            a = volume.numpy()
            if cnt <= 10:
                continue
            X.append(a)
            Y.append(label)
    except tf.errors.OutOfRangeError:
        return X, Y
        pass


def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)


s = StandardScaler()


def standardization(data, type):
    if type == 'valid':
        # 验证集 直接用transform
        return s.transform(
            data.astype(np.float32).reshape(-1, 1)).reshape(data.shape)
    else:
        return s.fit_transform(
            data.astype(np.float32).reshape(-1, 1)).reshape(data.shape)


def normalization(data):
    minVals = data.min()
    maxVals = data.max()
    ranges = maxVals - minVals
    normData = Normalizer().fit_transform(np.array(data, dtype='float32').reshape(1, -1)).reshape(data.shape)
    # normData = normData.astype(np.float32).reshape(data.shape)
    # normData = np.zeros(np.shape(data))
    # # m = normData.shape[0]
    # normData = data - np.tile(minVals, np.shape(data))
    # normData = normData / np.tile(ranges, np.shape(data))

    return normData, ranges, minVals


def create_model():
    input_img = keras.Input(shape=input_shape)
    latent = autoencoder(input_img)
    reconstruction = autodecoder(latent)
    model = keras.Model(input_img, reconstruction)
    # model = multi_gpu_model(model, GPU_COUNT)
    opt = keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=opt, loss=loss_type,
                  metrics=['accuracy'])
    model.summary()
    return model


# # cross validation
def split(train_X, i):
    data_size = len(train_X)
    group_size = int(data_size / n_splits)
    train = []
    valid = []
    for k in range(len(train_X)):
        if k in range(i * group_size, (i + 1) * group_size):
            valid.append(train_X[0])
        else:
            train.append(train_X[0])

    return train, valid


def main():
    # data loading
    # brain_word_data = {}
    brain_data = []
    label_data = []
    for user in participants:
        filenames = '/Storage/ying/resources/BrainBertTorch/dataset/alice/tf_data/test_data_vol_' + str(user) + '.tfrecords'
        # x.append -> x = {0:xx,1:xx,...}
        # x.extend -> x = {xx,xx,xx,xx,xx...}
        brain, label = generate_array_from_file(filenames)
        brain_data.extend(brain)
        label_data.extend(label)

    # data spilt
    brain_word_data = {'brain': brain_data, 'label': label_data}

    train_X, test_X, train_ground, text_ground = train_test_split(brain_word_data['brain'],
                                                                  brain_word_data['brain'],
                                                                  test_size=valid_train_rate,
                                                                  random_state=seed)
    auto_encoder = create_model()
    # cross validation
    for i in range(n_splits):
        train, valid = split(train_X, i)
        if i == n_splits:
            break
            # # 归一化  x = （x - u（均值）） / std(方差) 就得到符合均值是0,方差是1的
        train_x = np.array(train_X).reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1])
        train_x, _, _ = normalization(train_x)
        valid_x = np.array(test_X).reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1])
        valid_x, _, _ = normalization(valid_x)
        #
        #  正态分布
        x_train_scaled = standardization(train_x, 'train')
        # x_train_scaled, _, _ = normalization(x_train_scaled)
        x_vaild_scaled = standardization(valid_x, 'valid')
        # x_vaild_scaled, _, _ = normalization(x_vaild_scaled)
        autoencoder_train = auto_encoder.fit(x_train_scaled, train_x,
                                             batch_size=batch_size,
                                             epochs=n_epochs,
                                             shuffle=True,
                                             validation_data=(x_vaild_scaled, valid_x),
                                             verbose=1, callbacks=callbacks)
        # autoencoder_train = auto_encoder.fit(x_train_scaled, train_x,
        #                                      batch_size=batch_size,
        #                                      epochs=n_epochs,
        #                                      shuffle=True,
        #                                      validation_data=(x_vaild_scaled, valid_x),
        #                                      verbose=1, callbacks=callbacks)
        # Score trained model.
        loss = autoencoder_train.history['loss']
        val_loss = autoencoder_train.history['val_loss']
        accuracy = autoencoder_train.history['accuracy']
        val_accuracy = autoencoder_train.history['val_accuracy']
        loss_dic[i] = loss
        acc_dic[i] = accuracy
        vloss_dic[i] = val_loss
        vacc_dic[i] = val_accuracy
        # plt.figure()
        # plt.plot(np.arange(0, n_epochs), loss, linewidth='1', color='coral',
        #          marker='|', label='Training')
        # plt.plot(np.arange(0, n_epochs), val_loss, color='green', linestyle='dashed', marker=',',
        #          markerfacecolor='blue', label='Validation')
        # plt.title('Training and validation loss:epochs-' + str(n_epochs) + '-batch:' + str(batch_size))
        # plt.legend(loc='upper left')
        # plt.savefig(loss_img_path)
        # # plt.show()
        # # if loss_type == 'binary_crossentropy':
        # plt.figure()
        # plt.plot(np.arange(0, n_epochs), accuracy, linewidth='1', color='coral',
        #          marker='|', label='Training')
        # plt.plot(np.arange(0, n_epochs), val_accuracy, color='green', linestyle='dashed', marker=',',
        #          markerfacecolor='blue', label='Validation')
        # plt.title('Training and validation accuracy:epochs-' + str(n_epochs) + '-batch:' + str(batch_size))
        # plt.legend(loc='upper left')
        # plt.savefig(acc_img_path)

    auto_encoder.save(output_model_file)
    auto_encoder.save_weights(output_model_weight_file)

    # scores = auto_encoder.evaluate(x_vaild_scaled, valid_X, batch_size=batch_size, verbose=2)


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
    time_cost = str((end - start) / 60)
    print("time-cost:", time_cost)

    write_score(loss_dic, acc_dic, vloss_dic, vacc_dic, time_cost)
