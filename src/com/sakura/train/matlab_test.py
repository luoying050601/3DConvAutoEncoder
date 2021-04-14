import h5py
import numpy as np
from scipy import io
import pickle
import os

# 物語の名前のリスト
# 被験者に提示されたのが、read -> listen か listen -> read (train用
Storynames_RL = ["S00F0131", "S00F0374", "S00F1396", "S00M1046", "S01F1707",
                 "S02F0100", "S02F1704", "S02M1715", "S03F0072", "S04M0497",
                 "S04M0790", "S05M1110", "S06M0740", "S08M1702"]

# listen -> listen (test用
Storynames_LL = ['S02M1700', 'S02F1109']

# 物語の名前のリストを、resp.h5での保存名に変更する。
path_to_resp = 'resp.h5'

# 読む/聴くを1回ずつの物語を、聴いている際の脳画像データの名前のリスト
respnamelist_rl = ['Listen_' + thisstoryname for thisstoryname in Storynames_RL]

# 聴くを2回の物語を、聴いている際の脳画像データの名前のリスト（_2は2回めを指す）
respnamelist_ll = ['Listen_' + thisstoryname + '_2'*i for thisstoryname in Storynames_LL for i in range(2)]

# resp.h5の各データは、解析に使うとは限らない余分な領域を含みます。そこで、興味のある領域のvoxelを抜き出します。
# 各領域に所属するvoxelのインデックスは、その被験者のvset_xxx.matに格納されています。(xxxは数字)
# 例えば、全皮質の脳活動を抜き出す場合、99番のvsetに入っているインデックスを使えばよいです。
path_to_vset = 'vset_099.mat'

# matlabはインデックスが１始まりですが、pythonは0始まりです。
# vsetはmatlabのインデックスを想定しているため、pythonで利用する場合は、１を引いて辻褄を合わせる必要があります。
idx_voxel = io.loadmat(path_to_vset)['tvoxels'].squeeze() - 1

# 保存するディレクトリの指定&なければ作成
save_dir = "/"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)


# resp.h5から、興味のある領域の脳活動データを抜き出し、物語ごとに格納
resps = {}
with h5py.File(path_to_resp, 'r') as f:
    # 読む/聴くを一回ずつの物語を、読んでいる際の脳画像データの取り出し
    for n, thisrespname in enumerate(respnamelist_ll + respnamelist_rl):
        thisepi_shape = f[thisrespname].shape
        # (時間, (3次元)）となっているので、(時間, 1次元)に変形。
        thisepi = np.reshape(f[thisrespname], (thisepi_shape[0], np.prod(thisepi_shape[1:])))
        # 全皮質のボクセルを抜き出してrespsに格納
        resps[thisrespname] = thisepi[:, idx_voxel]
        print('response {} loaded.'.format(thisrespname))

        if (n == 0) or (n == 2):
            save_path = "{}resp_{}_1.pkl".format(save_dir, thisrespname)
        else:
            save_path = "{}resp_{}.pkl".format(save_dir, thisrespname)
        with open(save_path, "wb") as fs:
            pickle.dump(resps[thisrespname], fs)
        print(thisrespname, resps[thisrespname].shape)

# # test用に2回聴いたデータを平均
# for thisrespname in Storynames_LL:
#     save_path_1 = "{}resp_Listen_{}_1.pkl".format(save_dir, thisrespname)
#     save_path_2 = "{}resp_Listen_{}_2.pkl".format(save_dir, thisrespname)
#     with open(save_path_1, "rb") as fs_1:
#         resp_array_1 = pickle.load(fs_1)
#     with open(save_path_2, "rb") as fs_2:
#         resp_array_2 = pickle.load(fs_2)
#
#     resp_arv = (resp_array_1 + resp_array_2) / 2  # 平均
#     print(resp_arv.shape)
#
#     save_path_arv = "{}resp_Listen_{}.pkl".format(save_dir, thisrespname)
#     with open(save_path_arv, "wb") as fs_arv:
#         pickle.dump(resp_arv, fs_arv)
