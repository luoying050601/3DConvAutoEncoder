import os
import numpy as np
# import tensorflow as tf
import time
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import sys
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# acutally can go with GPU without adjustment
# print("****************************************************")
print(tf.test.is_gpu_available())

# hyperparameter definition
# (79, 95, 68, 372)
IMAGE_HEIGHT = 72
IMAGE_WIDTH = 96
IMAGE_DEPTH = 64
batch_size = 8  # >= 16 fail  CPU->GPU Memcpy failed
samples = 72 * 96 * 64
volume_shape = [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH]
input_shape = [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1]

n_epochs = 10
learning_rate = 0.001
noise_factor = 0.3
num_threads = 4
min_after_dequeue = 4000
capacity = 15000  # 15000
n_batches = int(samples / batch_size)
padding = 'SAME'
stride = [1, 1, 1]

DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../"))
# train
tfrecords_tr_vol_filename = DATA_DIR + '/output/fmri/tf_data/train_data_vol.tfrecords'
# validation
tfrecords_val_vol_filename = DATA_DIR + '/output/fmri/tf_data/val_data_vol.tfrecords'
# paths for saving summary and weights&biases.
DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../"))
logs_path = DATA_DIR + "/output/summary/"
ws_path = DATA_DIR + "/output/weights/"


# training related variables


# method to retrive a volume using the filename queue
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={'vol_raw': tf.FixedLenFeature([], tf.string)})
    vol_str = tf.decode_raw(features['vol_raw'], tf.float32)
    volume = tf.reshape(vol_str, volume_shape)
    volume_batch = tf.train.shuffle_batch([volume], batch_size=batch_size, capacity=capacity, num_threads=num_threads,
                                          min_after_dequeue=min_after_dequeue)
    finalbatch = tf.expand_dims(volume_batch, -1)

    return finalbatch
    # return volume


# problem: Input to reshape is a tensor with 491244(?confirm the vol data) values,
#  but the requested shape has 362880
# result : just because the data cut by wrong code.
# method to sample a batch using the input pipeline
def input_pipeline_tr():
    filename_queue = tf.train.string_input_producer(
        [tfrecords_tr_vol_filename], capacity=capacity, shuffle=True)
    # volume = read_and_decode(filename_queue)
    # volume_batch = tf.train.shuffle_batch([volume], batch_size=batch_size, capacity=capacity, num_threads=num_threads,
    #                                       min_after_dequeue=min_after_dequeue)
    # finalbatch = tf.expand_dims(volume_batch, -1)

    return read_and_decode(filename_queue)


# method to sample a batch using the input pipeline from the validation data
def input_pipeline_val():
    filename_queue = tf.train.string_input_producer([tfrecords_val_vol_filename], shuffle=True)
    # volume =

    return read_and_decode(filename_queue)


# the main method
def main():
    # start of tensorflow graph
    # input and target placeholders
    global nvol, batch_cost
    # print("input_shape:", input_shape)
    # inputs_ = tf.Variable(shape=input_shape, name="inputs")
    # targets_ = tf.Variable(shape=input_shape, name="targets")

    inputs_ = tf.placeholder(tf.float32, input_shape, name='inputs')
    targets_ = tf.placeholder(tf.float32, input_shape, name='targets')

    conv1 = tf.keras.layers.Conv3D(
        filters=16, kernel_size=(3, 3, 3), strides=stride, padding=padding, activation=tf.nn.relu)(inputs_)
    maxpool1 = tf.keras.layers.MaxPool3D(
        pool_size=(2, 2, 2), strides=(3, 2, 2), padding=padding)(conv1)
    # print('shape maxpool1:', maxpool1.shape)
    conv2 = tf.keras.layers.Conv3D(
        filters=32, kernel_size=(3, 3, 3), strides=stride, padding=padding, activation=tf.nn.relu)(maxpool1)
    maxpool2 = tf.keras.layers.MaxPool3D(
        pool_size=(2, 2, 2), strides=(3, 3, 2), padding=padding)(conv2)

    # print('shape:maxpool2', maxpool2.shape)
    conv3 = tf.keras.layers.Conv3D(
        filters=96, kernel_size=(2, 2, 2), strides=stride, padding=padding, activation=tf.nn.relu)(maxpool2)
    maxpool3 = tf.keras.layers.MaxPool3D(
        pool_size=(2, 2, 2), strides=(1, 1, 2), padding=padding)(conv3)
    # print('shape maxpool3:', maxpool3.shape)
    # decoder
    unpool1 = K.resize_volumes(maxpool3, 1, 1, 2, "channels_last")
    deconv1 = tf.keras.layers.Conv3DTranspose(filters=96, kernel_size=(2, 2, 2), strides=stride,
                                              padding=padding, activation=tf.nn.relu)(unpool1)
    # print('shape deconv1:', deconv1.shape)
    unpool2 = K.resize_volumes(deconv1, 3, 3, 2, "channels_last")
    deconv2 = tf.keras.layers.Conv3DTranspose(filters=32, kernel_size=(3, 3, 3), strides=stride,
                                              padding=padding, activation=tf.nn.relu)(unpool2)

    # print('shape deconv2:', deconv2.shape)
    # (64, 24, 48, 32, 32)
    unpool3 = K.resize_volumes(deconv2, 3, 2, 2, "channels_last")
    deconv3 = tf.keras.layers.Conv3DTranspose(filters=16, kernel_size=(3, 3, 3), strides=stride,
                                              padding=padding, activation=tf.nn.relu)(unpool3)

    # print('shape deconv3:', deconv3.shape)
    # (64, 72, 96, 64, 16)
    output = tf.keras.layers.Dense(
        units=1, activation=None)(deconv3)

    loss = tf.divide(tf.norm(tf.subtract(targets_, output), ord='fro', axis=[0, -1]),
                     tf.norm(targets_, ord='fro', axis=[0, -1]))
    # print(loss.shape)
    print("loss:", loss)
    cost = tf.reduce_mean(loss, name='loss')
    # print(cost)
    print("cost:", cost)
    opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    print("opt:", opt)

    all_saver = tf.train.Saver(max_to_keep=None)
    # conv1_v = tf.assign("conv1_v", conv1)
    # maxpool1_v = tf.assign("maxpool1_v", maxpool1)
    # conv2_v = tf.assign("conv2_v", conv2)
    # maxpool2_v = tf.assign("maxpool2_v", maxpool2)
    # conv3_v = tf.assign("conv3_v", conv3)
    # maxpool3_v = tf.assign("maxpool3_v", maxpool3)
    # enc_saver = tf.train.Saver({'conv1': conv1, 'maxpool1': maxpool1,
    #                             'conv2': conv2, 'maxpool2': maxpool2,
    #                             'conv3': conv3, 'maxpool3': maxpool3})
    # # initializing a saver to save weights
    # enc_saver = tf.train.Saver({'conv1': conv1_v, 'maxpool1': maxpool1_v,
    #                             'conv2': conv2_v, 'maxpool2': maxpool2_v,
    #                             'conv3': conv3_v, 'maxpool3': maxpool3_v})
    # initializing a restorer to restore weights
    # res_saver = tf.train.import_meta_graph('/weights/model.ckpt-1.meta')
    #
    # summary nodes
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("cost", cost)
    tf.summary.histogram("conv1", conv1)
    tf.summary.histogram("maxpool1", maxpool1)
    tf.summary.histogram("conv2", conv2)
    tf.summary.histogram("maxpool2", maxpool2)
    tf.summary.histogram("conv3", conv3)
    tf.summary.histogram("maxpool3", maxpool3)
    tf.summary.histogram("unpool3", unpool3)
    tf.summary.histogram("deconv3", deconv3)
    tf.summary.histogram("unpool2", unpool2)
    tf.summary.histogram("deconv2", deconv2)
    tf.summary.histogram("unpool1", unpool1)
    tf.summary.histogram("deconv1", deconv1)

    # summary operation and a writer to save it.
    summary_op = tf.summary.merge_all(key='summaries')
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # end of tensorflow graph

    # initializing tensorflow graph and a session
    init_op = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init_op)

    # making operation-variables to run our methods whenever needed during training
    fetch_op_tr = input_pipeline_tr()
    fetch_op_val = input_pipeline_val()

    # coordinator and queue runners to manage parallel sampling of batches from the input pipeline
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # start of training
    counter = 0
    try:

        while not coord.should_stop():
            print('\nEpoch\t' + str(counter + 1) + '/' + str(n_epochs))

            for i in range(n_batches):
                # fetching a batch
                vol = sess.run(fetch_op_tr)
                nvol = np.asarray(vol)
                noisy_nvol = nvol + noise_factor * np.random.randn(*nvol.shape)
                batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: noisy_nvol, targets_: nvol})
                if i % 1000 == 0:
                    print("batch_cost", batch_cost)
                print('\r' + str(((i + 1) * 100) / n_batches) + '%', sys.stdout.flush())
            counter = counter + 1
            print("Epoch: {}/{}...".format(counter, n_epochs), "Training loss: {:.4f}".format(batch_cost))
            print("time cost: {}".format(time.time()))
            # save weights and biases of the model
            all_saver.save(sess, ws_path + "model.ckpt", global_step=counter)
            # save weights and biases of the encoder
            # enc_saver.save(sess, ws_path + "enc.ckpt", global_step=counter)
            print('Weights saved')

            # saving summary  code above is clear
            # print(nvol.shape)
            # print(nvol.shape)
            # summary, _ = sess.run([summary_op, opt], feed_dict={inputs_: nvol, targets_: nvol})
            # print("summary:", summary)
            # print("counter:", counter)
            # writer.add_summary(summary, counter)
            print('Summary saved')

            if counter >= n_epochs:
                break
        # checking validation error
        vol = sess.run(fetch_op_val)
        nvol = np.asarray(vol)
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: nvol, targets_: nvol})
        print('Validation error' + str(batch_cost))
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

    # '''
    # code to restore weights
    with tf.Session(config=config) as sess:
        all_saver.restore(sess,  ws_path + "model.ckpt")
        print("Model restored.")
    # '''


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

    fileName = datetime.datetime.now().strftime('day and time:' + '%Y_%m_%d, %H:%M:%S')
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
    print((end - start) / 60)
