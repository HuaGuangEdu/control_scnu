# 作者：tomoya
# 创建：2022-09-29
# 更新：2022-09-29
# 用意：机器学习中调用的BP神经网络

from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import numpy as np
from matplotlib.animation import FuncAnimation

plt.rcParams['font.sans-serif'] = ['SimHei']
# 转换字典
# 各种转换字典
feature_names = {
    "iris": ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'],
    "digits": ["像素" + str(i) for i in range(64)],
    "wine": ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素', '颜色强度', '色调', 'od280/od315稀释葡萄酒', '脯氨酸'],
    "breast_cancer": [
        "半径（平均值）", "质地（平均值）", "周长（平均值）", "面积（平均值）", "光滑度（平均值）", "致密性（平均值）", "凹度（平均值）", "凹点（平均值）", "对称性（平均值）",
        "分形维数（平均值）", "半径（平均值）",
        "半径（标准差）", "质地（标准差）", "周长（标准差）", "面积（标准差）", "光滑度（标准差）", "致密性（标准差）", "凹度（标准差）", "凹点（标准差）", "对称性（标准差）",
        "分形维数（标准差）", "半径（标准差）",
        "半径（最大值）", "质地（最大值）", "周长（最大值）", "面积（最大值）", "光滑度（最大值）", "致密性（最大值）", "凹度（最大值）", "凹点（最大值）", "对称性（最大值）",
        "分形维数（最大值）", "半径（最大值）"
    ]
}
target_names = {
    "iris": ['山鸢尾', '变色鸢尾', '维吉尼亚鸢尾'],
    "digits": ["数字" + str(i) for i in range(10)],
    "wine": ["琴酒", "雪莉", "贝尔摩德"],
    "breast_cancer": ["阴性", "阳性"]
}
data_name = {
    "鸢尾花": "iris",
    "手写数字": "digits",
    "红酒": "wine",
    "威斯康辛州乳腺癌": "breast_cancer"
}
model_name = {
    "决策树": "1",
    "随机森林": "2",
    "支持向量机": "3",
    "神经网络": "4",
}
# 可视化部分函数
x, y = [], []  # 用于保存绘图数据，最开始时什么都没有，默认为空


def update(score):  # 更新函数
    x.append(len(y))  # 添加X轴坐标
    y.append(score)  # 添加Y轴坐标
    plt.plot(x, y, "r--")  # 绘制折线图


def tree_vis(model, dataName):
    fn = feature_names[dataName]
    cn = target_names[dataName]
    plt.figure("决策树或随机森林的可视化")
    plot_tree(model, filled=True, feature_names=fn, class_names=cn)


def studyVis(scores):
    fig = plt.figure("学习曲线(准确率)")
    plt.ylim(min(scores) * 0.999, max(scores) * 1.01)  # Y轴取值范围
    plt.ylabel("准确率", )  # Y轴刻度
    plt.xlim(0, len(scores) + 1)  # X轴取值范围
    plt.xlabel("训练轮数")  # Y轴刻度
    global x, y
    x, y = [], []  # 用于保存绘图数据，最开始时什么都没有，默认为空
    ani = FuncAnimation(fig, update, frames=scores, interval=3000 / len(scores), blit=False, repeat=False)  # 创建动画效果
    plt.show()


# 神经网络部分函数
def sigmod(z):
    h = 1. / (1 + np.exp(-z))
    return h


def de_sigmoid(z, h):
    return h * (1 - h)


def relu(z):
    h = np.maximum(z, 0)
    return h


def de_relu(z, h):
    z[z <= 0] = 0
    z[z > 0] = 1.0
    return z


def no_active(z):
    h = z
    return h


def de_no_active(z, h):
    return np.ones(h.shape)


# o Nxc
# lab Nxc
def loss_L2(o, lab):
    diff = lab - o
    sqrDiff = diff ** 2
    return 0.5 * np.sum(sqrDiff)


def de_loss_L2(o, lab):
    return o - lab


def loss_CE(o, lab):
    p = np.exp(o) / np.sum(np.exp(o), axis=1, keepdims=True)
    loss_ce = np.sum(-lab * np.log(p))
    return loss_ce


def de_loss_CE(o, lab):
    p = np.exp(o) / np.sum(np.exp(o), axis=1, keepdims=True)
    return p - lab


# dim_in:输入特征的维度
# list_num_hidden： 每层输出节点的数目
# list_act_funs： 每层的激活函数
# list_de_act_funs: 反向传播时的函数

def bulid_net(dim_in, list_num_hidden,
              list_act_funs, list_de_act_funs):
    layers = []

    # 逐层的进行网络构建
    for i in range(len(list_num_hidden)):
        layer = {}

        # 定义每一层的权重
        if i == 0:
            # layer["w"]= 0.2*np.random.randn(dim_in,list_num_hidden[i])-0.1 # 用sigmoid激活函数
            layer["w"] = 0.01 * np.random.randn(dim_in, list_num_hidden[i])  # 用relu 激活函数
        else:
            # layer["w"]= 0.2*np.random.randn(list_num_hidden[i-1],list_num_hidden[i])-0.1 # 用sigmoid激活函数
            layer["w"] = 0.01 * np.random.randn(list_num_hidden[i - 1], list_num_hidden[i])  # 用relu 激活函数

        # 定义每一层的偏置
        layer["b"] = 0.1 * np.ones([1, list_num_hidden[i]])
        layer["act_fun"] = list_act_funs[i]
        layer["de_act_fun"] = list_de_act_funs[i]
        layers.append(layer)

    return layers


# 返回每一层的输入
# 与最后一层的输出
def fead_forward(datas, layers):
    input_layers = []
    input_acfun = []
    for i in range(len(layers)):
        layer = layers[i]
        if i == 0:
            inputs = datas
            z = np.dot(inputs, layer["w"]) + layer["b"]
            h = layer['act_fun'](z)
            input_layers.append(inputs)
            input_acfun.append(z)
        else:
            inputs = h
            z = np.dot(inputs, layer["w"]) + layer["b"]
            h = layer['act_fun'](z)
            input_layers.append(inputs)
            input_acfun.append(z)
    return input_layers, input_acfun, h


# 进行参数更新更新
def updata_wb(datas, labs, layers, loss_fun, de_loss_fun, alpha=0.01):
    N, D = np.shape(datas)
    # 进行前馈操作
    inputs, input_acfun, output = fead_forward(datas, layers)
    # 计算 loss
    loss = loss_fun(output, labs)
    # 从后向前计算
    deltas0 = de_loss_fun(output, labs)
    # 从后向前计算误差
    deltas = []
    for i in range(len(layers)):
        index = -i - 1
        if i == 0:
            h = output
            z = input_acfun[index]
            delta = deltas0 * layers[index]["de_act_fun"](z, h)

        else:
            h = inputs[index + 1]
            z = input_acfun[index]
            delta = np.dot(delta, layers[index + 1]["w"].T) * layers[index]["de_act_fun"](z, h)

        deltas.insert(0, delta)

    # 利用误差 对每一层的权重进行修成
    for i in range(len(layers)):
        # 计算 dw 与 db
        dw = np.dot(inputs[i].T, deltas[i])

        db = np.sum(deltas[i], axis=0, keepdims=True)
        # 梯度下降
        layers[i]["w"] = layers[i]["w"] - alpha * dw
        # print(alpha * dw)
        # print("-" * 10)
        layers[i]["b"] = layers[i]["b"] - alpha * db
    return layers, loss


def test_accuracy(datas, labs_true, layers):
    _, _, output = fead_forward(datas, layers)
    lab_det = np.argmax(output, axis=1)
    labs_true = np.argmax(labs_true, axis=1)
    N_error = np.where(np.abs(labs_true - lab_det) > 0)[0].shape[0]

    error_rate = N_error / np.shape(datas)[0]
    return error_rate


def one_hot(target, classNum):
    N = len(target)
    lab_onehot = np.zeros([N, classNum])
    for i in range(N):
        id = int(target[i])
        lab_onehot[i, id] = 1
    return lab_onehot


class BPnet():
    def __init__(self):
        pass

    def __initNetStruct(self):
        list_num_hidden = [30, 5, self.classNum]
        list_act_funs = [relu, relu, no_active]
        list_de_act_funs = [de_relu, de_relu, de_no_active]
        # 定义损失函数
        self.loss_fun = loss_CE
        self.de_loss_fun = de_loss_CE
        self.model = bulid_net(self.featureNum, list_num_hidden,
                               list_act_funs, list_de_act_funs)

    def fit(self, x_train, y_train):
        self.train_data = normalize(x_train, axis=0, norm='max')
        self.classNum = np.max(y_train) + 1
        self.featureNum = x_train.shape[1]
        self.train_lab_onehot = one_hot(y_train, self.classNum)
        self.__initNetStruct()
        scores = []
        # 进行训练
        n_epoch = 5000

        batchsize = 20
        N = x_train.shape[0]
        N_batch = N // batchsize
        for i in range(n_epoch):
            # 数据打乱
            rand_index = np.random.permutation(N).tolist()
            # 每个batch 更新一下weight
            loss_sum = 0
            for j in range(N_batch):
                index = rand_index[j * batchsize:(j + 1) * batchsize]
                batch_datas = self.train_data[index]
                batch_labs = self.train_lab_onehot[index]
                layers, loss = updata_wb(batch_datas, batch_labs, self.model, self.loss_fun, self.de_loss_fun,
                                         alpha=0.01)
                loss_sum = loss_sum + loss

            error = test_accuracy(self.train_data, self.train_lab_onehot, self.model)
            score = 1 - error
            interval = n_epoch // 100
            if i % interval == 0:
                scores.append(score * 100)
        return scores

    def score(self, x_test, y_test):
        x_test = normalize(x_test, axis=0, norm='max')
        test_lab_onehot = one_hot(y_test, self.classNum)
        error = test_accuracy(x_test, test_lab_onehot, self.model)
        return (1 - error) * 100

    def predict(self, feature):
        feature = normalize(feature, axis=0, norm='max')
        _, _, output = fead_forward(feature, self.model)
        return np.argmax(output, axis=1)
