import numpy as np
import matplotlib.pyplot as mp
import time
import seaborn as sns


def generate_dataset(m1, m2, s1, s2, size, seed, split_ratio):
    np.random.seed(seed)
    # m1 = np.array([-1, 0])  # [-5, 0]
    # s1 = np.identity(2)
    # m2 = np.array([0, 1])  # [0, 5]
    # s2 = np.identity(2)  # 设置初始条件

    X1 = np.random.multivariate_normal(m1, s1, size).T  # 每一个x为列向量
    X2 = np.random.multivariate_normal(m2, s2, size).T
    y1 = np.ones((size, 1))  # y为列向量
    y2 = -np.ones((size, 1))  # 生成数据集x和标签y

    X = np.concatenate((X1, X2), axis=1)
    Y = np.concatenate((y1, y2), axis=0)

    # 随机打乱数据
    shuffle_indices = np.random.permutation(X.shape[1])
    X = X[:, shuffle_indices]
    Y = Y[shuffle_indices, :]

    # 生成训练集和测试集
    split_index = int(X.shape[1] * split_ratio)

    X_train = X[:, :split_index]
    Y_train = Y[:split_index]  # 前80%训练
    X_test = X[:, split_index:]
    Y_test = Y[split_index:]  # 后20%测试

    return X_train, Y_train, X_test, Y_test


def calcu_accuracy(w, x, y):
    num_wrong_arr = calcu_false(w, x, y)
    num_wrong, f = find_false(num_wrong_arr)
    return num_wrong


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def find_false(c):
    a = c
    f = list()
    num = 0
    y = len(a)
    for m in range(0, y):
        if a[m] != 0:
            num = num + 1
            f.append(m)
        else:
            continue
    if len(f) == 0:
        f = [0]
    return num, f

    # 比较是否出错


def calcu_false(w1, x, y):
    # x为列向量, y为列向量
    dot = np.dot(w1.T, x)
    dot[dot < 0] = -1
    dot[dot > 0] = 1
    num = dot.T - y
    return num


def oversize(x):
    # x为列向量
    m, n = np.shape(x)
    one_row = np.ones((1, n))
    x = np.vstack((one_row, x))
    return x


def output(num_train, num_test, size, split_ratio):
    total_train = size*2*split_ratio
    total_test = size*2 - total_train
    print(f'在训练集上的正确率是{((total_train - num_train) / total_train) * 100}%  '
          f'在测试集上的正确率是{((total_test - num_test) / total_test) * 100}%')

def plot(w, x, y, title):  # 传入矩阵形如（3， 200）（200， 1）
    # 画图
    sns.set_theme()
    sns.relplot(
        x=x[1, :], y=x[2, :], hue=y.flatten(), kind="scatter"
    )
    x_min1, x_max1 = x[1, :].min() - 1, x[1, :].max() + 1
    y_min1, y_max1 = x[2, :].min() - 1, x[2, :].max() + 1
    xx1, yy1 = np.meshgrid(np.arange(x_min1, x_max1, 0.01), np.arange(y_min1, y_max1, 0.01))
    z = np.dot(w[1:].T, np.vstack((xx1.ravel(), yy1.ravel())))
    z = z.reshape(xx1.shape)
    mp.contour(xx1, yy1, z, levels=[0], colors='b', linestyles='-')
    mp.title(title)
    mp.xlabel('X1')
    mp.ylabel('X2')
    mp.savefig(f'{title}.png', dpi=300)


