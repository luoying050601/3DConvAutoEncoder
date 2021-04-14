# useless
import SimpleITK as sitk
import time
import os
from multiprocessing import Pool
import tensorflow as tf

# tfrecords_filename = 'train_data.tfrecords'
# paths for saving summary and weights&biases.
DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../"))
# print(DATA_DIR)

# n_files = 23
# subjects = ['derivatives']
# val
participants = {1: ['18', '22', '23'],
                2: ['24', '26', '28'],
                3: ['30', '31', '35'],
                4: ['36', '37', '39'],
                5: ['41', '42', '43'],
                6: ['44', '45', '47'],
                7: ['48', '49'],
                8: ['50', '51', '53']
                }
# tfrecords_filename = DATA_DIR + '/output/tfrecord/sub-01.tfrecords'
# tfrecords_filename = DATA_DIR + '/output/tfrecord/sub-02.tfrecords'
# tfrecords_filename = DATA_DIR + '/output/tfrecord/sub-03.tfrecords'

# train
# participants = ['18', '22', '23', '24', '26', '28', '30', '31', '35', '36', '37', '39',
#                 '41', '42', '43', '44', '45', '47', '48', '49']
# tfrecords_filename = DATA_DIR + '/output/tfrecord/train_data.tfrecords'
# test
# participants = ['18']
file_name = {
    'derivatives': 'task-alice_bold_preprocessed',
    'func': 'task-alice_bold',
    'anat': 'T1w'
}


# writer = tf.python_io.TFRecordWriter(tfrecords_filename_val)


# some functions for tfrecords
def _bytes_feature(value):
    value = tf.compat.as_bytes(value)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def process(user):
    # for sub in subjects:
    #     print(user, sub)
    img_data = sitk.ReadImage(
        DATA_DIR + '/original_data/fMRI/sub-' + user + '/derivatives/sub-' + user + '_' + file_name[
            'derivatives'] + '.nii.gz')
    img_data = sitk.GetArrayFromImage(img_data)
    img_shape = img_data.shape
    img_raw = img_data.tostring()
    img_shape_raw = str(img_shape)
    example = tf.train.Example(features=tf.train.Features(
        feature={'img_raw': _bytes_feature(img_raw), 'img_shape': _bytes_feature(img_shape_raw)}))
    writer.write(example.SerializeToString())
    print('\nfile\t' + user + '\tadded to ' + tfrecords_filename + ' file')



if __name__ == '__main__':

    start = time.perf_counter()
    # print(i)

    # pool = Pool(3)
    # pool.map(process, participants.get(i), writer)
    for i in range(1, 9):
        tfrecords_filename = DATA_DIR + '/output/fmri/tf_data/S0' + str(i) + '.tfrecords'
        writer = tf.compat.v1.python_io.TFRecordWriter(tfrecords_filename)
        for user in participants.get(i):
            process(user)

        writer.close()
        print('\nThe file ' + tfrecords_filename + ' is saved in your home directory\n')

    # for i in range(1, 8):
    end = time.perf_counter()
    print((end - start) / 60)
