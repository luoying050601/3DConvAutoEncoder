# USELESS
import os
import torch
import time
import scipy.io as io
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot as plt
import tensorflow.keras.backend as K
from transformers import BertModel, BertTokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Flatten, Dense

batch_size = 2
LR = 0.0005
# n_splits = 10
n_epochs = 150
parallel_work = 1
IMAGE_HEIGHT = 72
IMAGE_WIDTH = 96
IMAGE_DEPTH = 64
volume_shape = [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH]
PROJ_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../"))
model_file = PROJ_DIR + '/output/weights/autoencoder_mri_kf.h5'
annotation_file = PROJ_DIR + '/output/annotation/annotations.tsv'
model_weight_file = PROJ_DIR + '/output/weights/autoencoder_weight_mri.h5'
loss_img_path = PROJ_DIR + "/output/images/L1_loss_" + str(n_epochs) + "_" + str(batch_size) + ".png"

# Create a description of the features.
feature_description = {
    'vol_raw': tf.io.FixedLenFeature([], tf.string)
    # 'sub-id': tf.io.FixedLenFeature([], tf.string)
}
output_model_file = PROJ_DIR + '/output/weights/L1_Model.h5'
LOG_DIR = PROJ_DIR + '/output/log'


def noramlization(data):
    minVals = data.min()
    maxVals = data.max()
    ranges = maxVals - minVals
    # normData = np.zeros(np.shape(data))
    # m = normData.shape[0]
    normData = data - np.tile(minVals, np.shape(data))
    normData = normData / np.tile(ranges, np.shape(data))
    return normData, ranges, minVals


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
            label = tf.compat.v1.decode_raw(parsed_record['label'], tf.string)
            volume = tf.reshape(feature, volume_shape)
            a = volume.numpy()
            if cnt <= 10:
                continue
            X.append(a)
            Y.append(label)
    except tf.errors.OutOfRangeError:
        return X,Y
        pass


def _parse_function(example_proto):
    # Parse the input `tf.Example` proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)


def L1_loss(y_true, y_pre):
    text_embedding = load_text_embedding_list()
    text_embedding, _, _ = noramlization(text_embedding)
    v = tf.reshape(tf.convert_to_tensor(text_embedding), [1, -1, 1024])

    return K.categorical_crossentropy(y_true, y_pre) + K.sum(K.abs(v - y_pre))


def dense_encoder(input_img):
    x = Dense(units=1024, input_dim=1024, kernel_initializer='normal', activation='relu')(input_img)
    # x = Dense(units=1024, kernel_initializer='normal', activation='relu')(x)
    # x = Dense(units=1024, kernel_initializer='normal', activation='relu')(x)
    # x = Dense(units=1024, kernel_initializer='normal', activation='relu')(x)
    # x = Dense(units=1024, kernel_initializer='normal', activation='relu')(x)
    # x = Dense(units=1024, kernel_initializer='normal', activation='relu')(x)
    x = Dense(units=1024, kernel_initializer='normal', activation='sigmoid')(x)
    # sigmoid good (5705 stop)
    # tanh 效果和sigmoid一毛一样
    # softmax 超垃圾
    # linear  效果和sigmoid一毛一样
    # elu -》5724
    # selu-》5726
    # relu  超垃圾

    return x


def main():
    model = load_model(model_file)
    # model.summary()
    # 新模型调用predict函数即可得到相当于原模型的中间层的输出结果
    embedding_test_data = PROJ_DIR + '/output/fmri/tf_data/test_data_vol_' + str(18) + '.tfrecords'
    test, label = generate_array_from_file(embedding_test_data)
    train_X = np.array(test).reshape([-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1])
    train_X, _, _ = noramlization(train_X)
    # (362,72,96,64,1)
    pretrained_model = keras.Model(inputs=model.input,
                                   outputs=model.get_layer("latent").output)  # 你创建新的模型
    encoded_imgs = pretrained_model.predict(train_X)
    encoded_imgs, _, _ = noramlization(encoded_imgs)
    z = tf.reshape(encoded_imgs, [1, -1, 1024])
    input_img = keras.Input(shape=[362, 1024])

    l1_model = keras.Model(input_img, dense_encoder(input_img))
    # model = keras.Sequential(extracted_layers)
    l1_model.summary()
    # K.constant(text_embedding)
    opt = keras.optimizers.Adam(learning_rate=LR)

    l1_model.compile(optimizer=opt,
                     loss=L1_loss,
                     metrics=['accuracy'])

    L1_train = l1_model.fit(z, z, batch_size=batch_size, shuffle=False, epochs=n_epochs)
    loss = L1_train.history['loss']
    plt.figure()
    plt.plot(np.arange(0, n_epochs), loss, linewidth='1', color='coral',
             marker='|', label='Training')
    plt.title('L1 loss:epochs-' + str(n_epochs) + '-batch:' + str(batch_size))
    plt.legend(loc='upper left')
    plt.savefig(loss_img_path)
    # plt.show()


def get_sentence_embedding(sentence, option):
    # if YOU NEED [cls] and [seq] PRESENTATION：True
    # --- load the model ----
    model = BertModel.from_pretrained('bert-large-uncased-whole-word-masking', output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')

    input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)

    all_layers_output = outputs[2]
    # list of torch.FloatTensor (one for the output of each layer
    # + the output of the embeddings) of shape (batch_size, sequence_length, hidden_size):
    # Hidden-states of the model at the output of each layer
    # plus the initial embedding outputs.

    if option == "last_layer":
        sent_embeddings = all_layers_output[-1]  # last layer
    elif option == "second_to_last_layer":
        sent_embeddings = all_layers_output[-2]  # second to last layer
    else:
        sent_embeddings = all_layers_output[-1]  # last layer

    sent_embeddings = torch.squeeze(sent_embeddings, dim=0)
    # print(sent_embeddings.shape)

    # Calculate the average of all token vectors.
    sentence_embedding_avg = torch.mean(sent_embeddings, dim=0)
    # print(sentence_embedding_avg.shape)

    return sent_embeddings.detach(), sentence_embedding_avg.detach()


def load_text_embedding_list():
    data = io.loadmat('text_embedding.mat')
    return data['result']


# once created it, this function is useless.
def create_text_embedding_list():
    alice_df = pd.read_csv(annotation_file, sep='\t', header=0)
    alice_sent = alice_df['Sentences']
    # alice_df['sentence_tensor'] = 0
    embedding_list = []
    text_list = []

    for i in range(len(alice_sent)):
        bert_layer_option = "last_layer"
        sent_embeddings, sent_embedding_avg = get_sentence_embedding(alice_sent[i], bert_layer_option)
        sub_np1 = sent_embedding_avg.numpy()
        embedding_list.append(sub_np1)
        text_list.append(alice_sent[i])
    io.savemat('text_embedding.mat', {'result': np.array(embedding_list), 'text': np.array(text_list)})


if __name__ == '__main__':
    start = time.perf_counter()
    # make_print_to_file(path='.')
    # create_text_embedding_list()
    main()
    end = time.perf_counter()
    print("time-cost:", str((end - start) / 60))
