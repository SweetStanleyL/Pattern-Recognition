import numpy as np
import matplotlib.pyplot as mp
import seaborn as sns
from function import generate_dataset, calcu_accuracy, sign, find_false, calcu_false, plot, output, oversize

# Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 逻辑回归函数
def logistic_regression(x, y, w, epochs, batch_size, lr=0.01):

    w_1 = w
    # 输入项说明：x为列向量，y为列向量，w为列向量
    y = y.T

    # 初始化矩阵大小
    m, n = np.shape(x)

    # 合并x,y矩阵
    z = np.concatenate((x, y), axis=0) # x为列向量，y为行向量

    # 根据batchsize确定迭代次数
    iterations = int(n / batch_size)

    for epoch in range(epochs):
        # 随机打乱数据
        shuffle_indices = np.random.permutation(n)
        z = z[:, shuffle_indices]
        for i in range(iterations):

            # 初始化梯度
            sum_gradient = np.zeros((m, 1))

            for j in range(batch_size):
                x1 = z[:m, i * batch_size + j].reshape(m, 1)
                y1 = z[m:, i * batch_size + j]

                # 计算预测值
                num_dot = np.dot(w_1.T, x1)
                # y_hat = sigmoid(num_dot)

                # 计算交叉熵损失梯度
                gradient = sigmoid(-y1*num_dot)*(-y1*x1).reshape(m, 1)
                sum_gradient = sum_gradient + gradient 

            # 更新权重
            avg_gradient = sum_gradient / batch_size
            w_1 = w_1 - lr * avg_gradient
            
    return w_1


 
def main():

    w_ini = np.array(([0.],
                  [1.],
                  [1.]))
    
    # 超参数
    size = 200
    seed = 0
    split_ratio = 0.8
    eta = 0.05
    epochs = 10
    batch_size = 150
    l_in = []

    # 生成数据集
    X_train, Y_train, X_test, Y_test = generate_dataset(np.array([-5, 0]), np.array([0, 5]),
                                                        np.identity(2), np.identity(2), 
                                                        size, seed, split_ratio)

    # 增加偏置项
    X_train = oversize(X_train)
    X_test = oversize(X_test)

    # 运行逻辑回归
    w = logistic_regression(X_train, Y_train, w_ini, epochs, batch_size, eta)

    # 计算出错数
    num_wrong_train = calcu_accuracy(w, X_train, Y_train)
    num_wrong_test = calcu_accuracy(w, X_test, Y_test)

    # 计算正确率
    output(num_wrong_train, num_wrong_test, size, split_ratio)

    # 画出分类面
    plot(w, X_train, Y_train, title='Gradient_descent_train')
    plot(w, X_test, Y_test, title='Gradient_descent_test')

    # 计算sigmoid预测值
    p_test = sigmoid(np.dot(w.T, X_test))
    p_test[p_test < 0.5] = 1 - p_test[p_test < 0.5]
    print(p_test)

    # 损失函数随epoch的变化
    for i in range(1, 50, 1):
        w_lin = logistic_regression(X_train, Y_train, w_ini, i, batch_size, eta)
        l_in_single = 1/X_test.shape[1] * np.sum(np.log(1 + np.exp(-np.dot(w_lin.T, X_test) * Y_test.T )))
        l_in.append(l_in_single)

    # 绘制损失函数随 epoch 变化的图像
    epoches = range(1, 50, 1)
    mp.figure()
    sns.set_theme()
    sns.lineplot(x=epoches, y=l_in)
    sns.scatterplot(x=epoches, y=l_in)
    mp.title('L_in-Epoch')
    mp.xlabel('Epoch')
    mp.ylabel('l_in')
    mp.grid(True)
    mp.savefig('L_in-Epoch.png', dpi=300)

    # mp.show()
    
if __name__ == "__main__":
    main()