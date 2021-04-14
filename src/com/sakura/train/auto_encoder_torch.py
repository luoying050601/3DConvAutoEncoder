import torch
from torch import nn, optim
import torch.utils.data as Data
# from torch.autograd import Variable
import numpy as np
import os
import time
from tfrecord.torch.dataset import TFRecordDataset

batch_size = 32
n_epochs = 10
LR = 0.005
stride = [1, 1, 1]
padding = [1, 1, 1]
IMAGE_HEIGHT = 72
IMAGE_WIDTH = 96
IMAGE_DEPTH = 64
volume_shape = [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH]
input_shape = [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 4]
dropout_flag = True
shuffle_flag = True
parallel_work = 1
DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../../../"))
tfrecord_path = DATA_DIR + '/output/fmri/tf_data/XX.tfrecords'


# tfrecord_path = "test_file"


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        # define encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=72, out_channels=72, kernel_size=(3, 3, 3), stride=stride, padding=padding),
            nn.Tanh(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(3, 2, 2), padding=padding),
            nn.Tanh(),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=stride, padding=padding),
            nn.Tanh(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(3, 3, 2), padding=padding),
            nn.Tanh(),
            nn.Conv3d(in_channels=96, out_channels=96, kernel_size=(3, 3, 3), stride=stride, padding=padding),
            nn.Tanh(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 1, 2), padding=padding),
            nn.Tanh()
        )
        # define decoder
        self.decoder = nn.Sequential(
            nn.MaxUnpool3d(kernel_size=(2, 2, 2), stride=(1, 1, 2), padding=padding),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels=96, out_channels=96, kernel_size=(3, 3, 3), stride=stride, padding=padding),
            nn.Tanh(),
            nn.MaxUnpool3d(kernel_size=(2, 2, 2), stride=(3, 3, 2), padding=padding),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=stride, padding=padding),
            nn.Tanh(),
            nn.MaxUnpool3d(kernel_size=(2, 2, 2), stride=(3, 2, 2), padding=padding),
            nn.Tanh(),
            nn.ConvTranspose3d(in_channels=16, out_channels=16, kernel_size=(3, 3, 3), stride=stride, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder
        # reconstructed = torch.relu(encoder)
        # return reconstructed


# load and normalize dataset
def get_data(img):
    return Data.DataLoader(dataset=img, shuffle=shuffle_flag, batch_size=batch_size,
                           drop_last=dropout_flag, num_workers=parallel_work)


def feature_normalize(data):
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mu) / std


def input_pipeline(input):
    # input.size() 64 1769472
    volume = torch.reshape(input, input_shape)

    return volume


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    auto_encoder = AE().to(device)
    optimizer = torch.optim.Adam(auto_encoder.parameters(), lr=LR, weight_decay=0.01)
    loss_func = nn.MSELoss()
    # load input data
    # index_path = None
    description = {"vol_raw": "byte"}
    test_dataset = TFRecordDataset(tfrecord_path, index_path=None, description=description)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    # data = next(iter(loader))
    # print(data)
    for epoch in range(n_epochs):
        loss = 0
        for step, input in enumerate(test_loader):
            # ===================== forward =========================
            input_x = input_pipeline(input['vol_raw'].float())
            encoder_out, decoder_out = auto_encoder(input_x)
            # encoder_out = encoder_out.float()
            decoder_out = decoder_out.float()
            # loss = loss_func(decoder_out, brain_img.float())
            # print(decoder_out)
            # print(decoder_out)

            # ===================== backward =========================
            optimizer.zero_grad()
            # loss.backward()
            optimizer.step()

            # --------------------accuracy begin-------------------------#
            # _, prediction = torch.max(decoder_out, 1)  # prediction里面是一维索引
            pred = decoder_out.argmax(dim=1)

            # correct_num += torch.eq(pred, brain_labels).sum().float().item()

            # print(prediction.shape)
            # print(prediction)
            # print(brain_labels.shape)
            # print(brain_labels)
            # correct += (prediction == brain_labels).sum().item()  # 获得一个batch正确的数量
            # out_class = (decoder_out[:] > 0).float()  # 将out矩阵中大于0的转化为1，小于0的转化为0，存入a中
            # right_num = torch.sum(b_x == out_class).float()  # 分类对的数值
            # precision = correct / decoder_out.shape[0]  # 准确率
            # --------------------accuracy end-------------------------#

            if step % 100 == 0:
                print('Epoch:{}, Train_loss:{:.8f}'.format(epoch, loss.item()))
                # print('Epoch:{}, Train_loss:{:.8f},precision:{:.8f}'.format(epoch, loss.item(), precision))
                # loss_list.append(loss.item())
                # Accuracy_list.append(precision)
    #
    # plt.plot(Epoch_line, loss_list, '.-', c='red', label='Loss change')
    # # plt.plot(Epoch_line, Accuracy_list, c='blue', label='precision')
    # plt.xticks(Epoch_line)
    # plt.xlabel('epoch')
    # plt.title('batch_size:' + str(Batch_size) + ' epoch:' + str(Epoch))
    # plt.legend()
    # plt.savefig(DATA_DIR + '/output/Loss_brain/result_' + str(int(time.time())) + '.jpg')
    # plt.show()
    # Update the lr for the next step
    # lr *= mult
    # optimizer.param_groups[0]['lr'] = lr


# return log_lrs, losses


if __name__ == '__main__':
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print((end - start) / 60)

# if __name__ == '__main__':
#     #  use gpu if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # create a model from `AE` autoencoder class
#     # load it to the specified device, either gpu or cpu
#     model = AE().to(device)
#
#     # create an optimizer object
#     # Adam optimizer with learning rate 1e-3
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#
#     # mean-squared error loss
#     criterion = nn.MSELoss()
#
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
#     )
#
