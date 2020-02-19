import numpy as np

# irisデータセットの読み込み
#   num_train: 学習データ数(残りはテストデータ)
#   random: ランダムに抽出するか
def load_iris(num_train=100, random=True):

    from sklearn.datasets import load_iris
    iris = load_iris()
    data = iris.data.astype(np.float32)
    label = iris.target.astype(np.int64)

    if random:
        perm = np.random.permutation(data.shape[0])
        a = perm[0:num_train]
        b = perm[num_train:]
    else:
        number = [i for i in range(len(data))]
        a = number[0:num_train]
        b = number[num_train:]

    train_data = data[a, :]
    train_teach = label[a]
    test_data = data[b, :]
    test_teach = label[b]

    return train_data, train_teach, test_data, test_teach

