import os
import sys
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
from keras import backend as K

# 動かない
# tf.compat.v1.disable_eager_execution()
# tf.debugging.set_log_device_placement(True)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # Restrict TensorFlow to only use the first GPU
#     try:
#         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#     except RuntimeError as e:
#         # Visible devices must be set before GPUs have been initialized
#         print(e)

print(tf.__version__)
# import tensorflow.compat.v1 as tf
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# tf_config = tf.compat.v1.ConfigProto()
# tf_config.gpu_options.allocator_type = 'BFC'
# tf_config.gpu_options.allow_growth = True
# device_name = '/device:GPU:1'
# allow_soft_placement=True,
# tf_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)

# latent representation :(8, 16, 8, 372)
# hyper parameter definition
IMAGE_HEIGHT = 72
IMAGE_WIDTH = 96
IMAGE_DEPTH = 64
batch_size = 16
volume_shape = [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH]
input_shape = [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1]
n_epochs = 10
# variable pertaining to the input pipeline
samples = 84 * 60 * 72
num_threads = 2
min_after_dequeue = 4000
capacity = 15000  # 15000
DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../"))

# paths for saving summary and weights&biases.
logs_path = DATA_DIR + "/output/summary/"
ws_path = DATA_DIR + "/output/weights/"

# training related variables
n_batches = int(samples / batch_size)
padding = 'SAME'
stride = [1, 1, 1]
learning_rate = 0.001
noise_factor = 0.3


# method to retrive a volume using the filename queue
def read_and_decode(filename_queue):
    # with tf.device(device_name):  # 指定特定device
    reader = tf.compat.v1.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    features = tf.compat.v1.parse_single_example(serialized_example,
                                                 features={'vol_raw': tf.compat.v1.FixedLenFeature([], tf.string)})
    vol_str = tf.compat.v1.decode_raw(features['vol_raw'], tf.float32)
    volume = tf.reshape(vol_str, volume_shape)
    return volume


# problem: Input to reshape is a tensor with 491244(?confirm the vol data) values,
#  but the requested shape has 362880
# result : just because the data cut by wrong code.
# method to sample a batch using the input pipeline
def input_pipeline(tfrecords_tr_filename):
    # with tf.device(device_name):  # 指定特定device
    filename_queue = tf.compat.v1.train.string_input_producer(
        [tfrecords_tr_filename], capacity=capacity, shuffle=True)
    volume = read_and_decode(filename_queue)
    volume_batch = tf.compat.v1.train.shuffle_batch([volume], batch_size=batch_size, capacity=capacity,
                                                    num_threads=num_threads,
                                                    min_after_dequeue=min_after_dequeue)
    final_batch = tf.expand_dims(volume_batch, -1)

    return final_batch


# method to sample a batch using the input pipeline from the validation data
# def input_pipeline_val():
#     filename_queue = tf.train.string_input_producer([tfrecords_val_vol_filename], shuffle=True)
#     volume = read_and_decode(filename_queue)
#     volume_batch = tf.train.shuffle_batch([volume], batch_size=batch_size, capacity=capacity, num_threads=num_threads,
#                                           min_after_dequeue=min_after_dequeue)
#     finalbatch = tf.expand_dims(volume_batch, -1)
#
#     return finalbatch


# the main method
def main():
    # mirrored_strategy = tf.distribute.MirroredStrategy()

    # with mirrored_strategy.scope():
    # with tf.compat.v1.Session(config=tf_config) as sess:
    tfrecords_test_vol_filename = DATA_DIR + '/output/fmri/tf_data/XXX.tfrecords'
    # train
    # tfrecords_tr_vol_filename = DATA_DIR + '/output/fmri/tf_data/train_data_vol.tfrecords'
    # validation
    # tfrecords_val_vol_filename = DATA_DIR + '/output/fmri/tf_data/val_data_vol.tfrecords'

    global batch_cost
    inputs_ = tf.compat.v1.placeholder(tf.float32, input_shape, name='inputs')
    targets_ = tf.compat.v1.placeholder(tf.float32, input_shape, name='targets')

    # encoder

    print('shape input:', inputs_.shape)
    conv1 = tf.keras.layers.Conv3D(
        filters=16, kernel_size=(3, 3, 3), strides=stride, padding=padding, activation=tf.nn.relu)(inputs_)
    maxpool1 = tf.keras.layers.MaxPool3D(
        pool_size=(2, 2, 2), strides=(3, 2, 2), padding=padding)(conv1)
    print('shape maxpool1:', maxpool1.shape)
    conv2 = tf.keras.layers.Conv3D(
        filters=32, kernel_size=(3, 3, 3), strides=stride, padding=padding, activation=tf.nn.relu)(maxpool1)
    maxpool2 = tf.keras.layers.MaxPool3D(
        pool_size=(2, 2, 2), strides=(3, 3, 2), padding=padding)(conv2)

    print('shape:maxpool2', maxpool2.shape)
    conv3 = tf.keras.layers.Conv3D(
        filters=96, kernel_size=(2, 2, 2), strides=stride, padding=padding, activation=tf.nn.relu)(maxpool2)
    maxpool3 = tf.keras.layers.MaxPool3D(
        pool_size=(2, 2, 2), strides=(1, 1, 2), padding=padding)(conv3)
    print('shape maxpool3:', maxpool3.shape)
    # decoder
    unpool1 = K.resize_volumes(maxpool3, 1, 1, 2, "channels_last")
    deconv1 = tf.keras.layers.Conv3DTranspose(filters=96, kernel_size=(2, 2, 2), strides=stride,
                                              padding=padding, activation=tf.nn.relu)(unpool1)
    print('shape deconv1:', deconv1.shape)
    unpool2 = K.resize_volumes(deconv1, 3, 3, 2, "channels_last")
    deconv2 = tf.keras.layers.Conv3DTranspose(filters=32, kernel_size=(3, 3, 3), strides=stride,
                                              padding=padding, activation=tf.nn.relu)(unpool2)

    print('shape deconv2:', deconv2.shape)
    # (64, 24, 48, 32, 32)
    unpool3 = K.resize_volumes(deconv2, 3, 2, 2, "channels_last")
    deconv3 = tf.keras.layers.Conv3DTranspose(filters=16, kernel_size=(3, 3, 3), strides=stride,
                                              padding=padding, activation=tf.nn.relu)(unpool3)

    print('shape deconv3:', deconv3.shape)
    # (64, 72, 96, 64, 16)
    output = tf.keras.layers.Dense(
        units=1, activation=None)(deconv3)
    print(output.shape)
    output = tf.reshape(output, input_shape)
    loss = tf.divide(tf.norm(tf.subtract(targets_, output), ord='fro', axis=[1, 2]),
                     tf.norm(targets_, ord='fro', axis=[1, 2]))

    # loss = tf.divide(tf.norm(tf.subtract(targets_, output), ord='fro', axis=[-2, -1]),
    #                  tf.norm(targets_, ord='fro', axis=[-2, -1]))
    cost = tf.reduce_mean(loss)
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cost)

    # output shape = input shape

    all_saver = tf.compat.v1.train.Saver(max_to_keep=None)

    # initializing a saver to save weights
    # enc_saver = tf.compat.v1.train.Saver({'conv1': conv1, 'maxpool1': maxpool1,
    #                             'conv2': conv2, 'maxpool2': maxpool2, 'conv3': conv3, 'maxpool3': maxpool3})
    # initializing a restorer to restore weights
    # res_saver = tf.compat.v1.train.import_meta_graph(ws_path+'weights.meta')

    # summary nodes
    # tf.summary.scalar("loss", loss)
    tf.summary.scalar("cost", cost)
    tf.summary.histogram("conv1", conv1)
    # tf.summary.histogram("conv1_1", conv1_1)
    tf.summary.histogram("maxpool1", maxpool1)
    tf.summary.histogram("conv2", conv2)
    tf.summary.histogram("maxpool2", maxpool2)
    tf.summary.histogram("conv3", conv3)
    tf.summary.histogram("maxpool3", maxpool3)
    # tf.summary.histogram("conv4", conv4)
    # tf.summary.histogram("deconv4", deconv4)
    tf.summary.histogram("unpool3", unpool3)
    tf.summary.histogram("deconv3", deconv3)
    tf.summary.histogram("unpool2", unpool2)
    tf.summary.histogram("deconv2", deconv2)
    tf.summary.histogram("unpool1", unpool1)
    # tf.summary.histogram("deconv1_1", deconv1_1)
    tf.summary.histogram("deconv1", deconv1)

    # summary operation and a writer to save it.
    summary_op = tf.compat.v1.summary.merge_all()
    writer = tf.compat.v1.summary.FileWriter(logs_path, graph=tf.compat.v1.get_default_graph())

    # end of tensorflow graph

    # start of training
    counter = 0
    try:
        sess = tf.compat.v1.Session()
        # making operation-variables to run our methods whenever needed during training
        # coordinator and queue runners to manage parallel sampling of batches from the input pipeline
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
        while not coord.should_stop():
            # fetching a batch
            fetch_op_tr = input_pipeline(tfrecords_test_vol_filename)
            nvol = sess.run(fetch_op_tr)

            batch_cost, _ = sess.run([cost, opt],
                                     feed_dict={inputs_: nvol + noise_factor * np.random.randn(*nvol.shape),
                                                targets_: nvol})
            # loss function, optimizer and a saver to save weights&biases

            print('\nEpoch\t' + str(counter + 1) + '/' + str(n_epochs))
            # # print(n_batches) 5670
            for i in range(n_batches):
                # if i % 1000 == 0:
                #     print("batch_cost:", batch_cost)
                print('\r' + str(((i + 1) * 100) / n_batches) + '%', sys.stdout.flush())
            counter = counter + 1
            # print("Epoch: {}/{}...".format(counter, n_epochs), "Training loss: {:.4f}".format(batch_cost))
            # save weights and biases of the model
            all_saver.save(sess, ws_path + "model.ckpt", global_step=counter)
            # save weights and biases of the encoder
            # enc_saver.save(sess, ws_path + "enc.ckpt", global_step=counter)
            print('Weights saved')

            # saving summary
            # summary, _ = sess.run([summary_op, opt], feed_dict={inputs_: nvol, targets_: nvol})
            # writer.add_summary(summary, counter)
            print('Summary saved')

            if counter >= n_epochs:
                break
        # checking validation error
        # vol = sess.run(fetch_op_val)
        # nvol = np.asarray(vol)
        # batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: nvol, targets_: nvol})
        # print('Validation error' + str(batch_cost))
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

    # with tf.compat.v1.Session() as s:
    #     all_saver.restore(s, "model.ckpt")
    #     print("Model restored.")
    #
    # '''
    #     #code to restore weights
    #
    #     '''


# start of tensorflow graph
# input and target placeholders
# print(input_shape)
# inputs_ = tf.Variable(shape=input_shape, name="inputs")
# targets_ = tf.Variable(shape=input_shape, name="targets")


def make_print_to_file(path='./'):
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

    fileName = datetime.datetime.now().strftime('day' + '%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)

    #############################################################
    # print -> log
    #############################################################
    print(fileName.center(60, '*'))


if __name__ == '__main__':
    # for d in ['/device:GPU:0', '/device:GPU:1', '/device:GPU:2', '/device:GPU:3']:
    # with tf.device('/device:GPU:1'):
    make_print_to_file(path='.')
    main()
