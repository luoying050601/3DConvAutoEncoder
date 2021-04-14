import os
import numpy as np
import nibabel as nib
import tensorflow as tf
# import pickle
import pandas as pd


# some functions for tfrecords
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# def _int64_feature(value):
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# def _float_feature(value):
#     return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))


# string variables needed to access tfrecords files
# , '22', '23', '24', '26', '28', '30', '31', '35', '36', '37', '39',
#                 '41', '42', '43', '44', '45', '47', '48', '49', '50', '51', '53'
DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../"))
participants = ['18']
# tfrecords_vol_filename = DATA_DIR + '/output/fmri/tf_data/train_data_vol.tfrecords'
anotation_file = DATA_DIR + '/output/annotation/annotations.tsv'

subjects = ['derivatives']
file_name = {
    'derivatives': 'task-alice_bold_preprocessed',
    'func': 'task-alice_bold',
    'anat': 'T1w'
}

# keep track of number of volumes in each tfrecord file.
# volPerFile = open("../volPerFile.txt", 'wb')

print("\nStarting conversion and masking...\n")

# for i in range(1, 2):
for user in participants:
    tfrecords_vol_filename = DATA_DIR + '/output/fmri/tf_data/test_data_vol_' + str(user) + '.tfrecords'
    writer = tf.compat.v1.python_io.TFRecordWriter(tfrecords_vol_filename)
    print(user)
    # create a tfrecords file
    # create a writer to write in it.

    # keep count of volumes
    vol_counter = 0
    func_filename = DATA_DIR + '/original_data/fMRI/' \
                               'sub-' + user + '/derivatives/sub-' + user + '_task-alice_bold_preprocessed.nii.gz'
    mask_filename = DATA_DIR + '/original_data/fMRI/sub-' + user + '/derivatives/sub-' + user + \
                    '_task-alice_bold_preprocessed_mask.nii.gz'
    # mask_filename = DATA_DIR + '/original_data/fMRI/sub-' + user + '/derivatives/aseg.nii.gz'

    img = nib.load(func_filename)
    img_data = img.get_fdata()
    # some stuff needed for next loop.
    img_shape = img_data.shape
    # print(img_shape)
    img_shape = np.asarray(img_shape)
    W = img_shape.item(3)
    # print(type(W))
    I32 = img_data.astype(np.float32)

    mask = nib.load(mask_filename)
    mask_data = mask.get_fdata()
    print(mask_data.shape)
    mask_data32 = mask_data.astype(np.float32)
    alice_df = pd.read_csv(anotation_file, sep='\t', header=0)
    alice_sent = alice_df['Sentences']
    vox = 0
    # for each volume in a file
    for vol in range(W):
        if vol > 9:
            volume = I32[:, :, :, vol]

            # mask the volumes
            volume = np.multiply(volume, mask_data32)
            # (79, 95, 68, 372)
            # (72-, 96+, 64-, 372)

            # crop 1 from each dimension
            volume = np.delete(volume, slice(72, 79), 0)
            volume = np.insert(volume, 95, 0, 1)
            # volume = np.delete(volume, slice(84, 95), 1)
            volume = np.delete(volume, slice(64, 68), 2)

            # flatten and write to tf file
            vol_str = volume.tostring()
            # vol_str = volume.astype(float)
            # print(vol_str[0])
            # print("*******")
            # print(vol-10)
            # print(alice_sent[vol-10])

            example = tf.train.Example(
                features=tf.train.Features(feature={'vol_raw': _bytes_feature(vol_str),
                                                    'sub-id': _bytes_feature(str.encode(user)),
                                                    'label': _bytes_feature(str.encode(alice_sent[vol - 10]))
                                                    }))
            writer.write(example.SerializeToString())
            threshold = 0
            volume[volume < threshold] = 0
            volume[volume > threshold] = 1
            num = volume.sum()
            print(num)
            vox = vox + num

            # print(user)
            # with open(pickle_vol_filename, 'wb') as f:
            #     pickle.dump(example.SerializeToString(), f)
            # keep track of number of volumes added to the tf file.
            vol_counter = vol_counter + 1

    print(vol_counter)
    print(vox)

    # close the file writer
    writer.close()
    # volPerFile.write(str(vol_counter).strip().encode('utf-8') + b"\n")

# keep track of number of volumes
# volPerFile.close()
