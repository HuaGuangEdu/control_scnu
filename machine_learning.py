import os
import joblib
from typing import Any
import math
from sklearn.model_selection import train_test_split
from .util.ml_tool import data_name, model_name, tree_vis, studyVis
from .util.all_path import model_path


class ModelNew:
    def __init__(self, name: str):
        """
        初始化模型的类
        :param name: 模型名字
        """
        self.model_name = name
        self.model = self.load(name)
        self.test_score = -1
        self.pred = -1

    def load(self, name: str):
        """
        导入模型，支持导入之前保存的模型，或者sklearn的模型
        :param name: 模型名字
        :return:
        """
        if name.endswith(".proto") or name in [i.replace(".proto", "") for i in os.listdir(model_path) if
                                               i.endswith(".proto")]:
            name = name + ".proto" if ".proto" not in name else ""
            if not os.path.isabs(name):
                name = os.path.join(model_path, name)
            model = joblib.load(name)

        elif name == model_name['决策树']:
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier()
        elif name == model_name['随机森林']:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, oob_score=True)

        elif name == model_name['支持向量机']:
            from sklearn.svm import SVC
            model = SVC()
        elif name == model_name['神经网络']:
            from .util.ml_tool import BPnet
            model = BPnet()
        else:
            raise ValueError("没有这个模型")
        return model

    def train(self, x_train, y_train, visible=False, dataName=''):
        """
        训练模型
        :param x_train: 数据集的特征
        :param y_train: 数据集的标签
        :param visible: 是否可视化训练过程
        :param dataName: 数据集的名字
        :return:
        """
        scores = []
        if self.model_name == model_name["神经网络"]:
            scores = self.model.fit(x_train, y_train)
        else:
            for i in range(math.ceil(len(y_train) / 15)):
                try:
                    self.model.fit(x_train[:(i + 1) * 10], y_train[:(i + 1) * 10])
                except:
                    self.model.fit(x_train, y_train)
                scores.append(self.model.score(x_train, y_train))
        if visible:
            if self.model_name == model_name['决策树']:
                tree_vis(self.model, dataName)

            elif self.model_name == model_name["随机森林"]:
                tree_vis(self.model.estimators_[0], dataName)
            studyVis(scores)

    def test(self, x_test, y_test):
        """
        测试模型
        :param x_test: 测试集特征
        :param y_test: 测试集标签
        :return:
        """
        self.test_score = self.model.score(x_test, y_test)

    def predict(self, feature):
        """
        使用训练完的模型进行预测
        :param feature: 输入的特征
        :return:
        """
        self.pred = self.model.predict(feature)

    def save(self, name: str):
        """
        保存模型
        Args:
            name: 保存的模型的名字，需要以pkl为后缀

        Returns:

        """
        if not name.endswith(".proto"):
            name += ".proto"
        try:
            saveName = (model_path if os.path.isabs(name) == False else "") + name
            joblib.dump(self.model, saveName)  # 模型后缀名统一为.proto
            print("保存模型成功！")
        except:
            print('保存模型失败！')

    def setParameter(self, **para: Any):
        """
        为各种模型设置参数
        Args:
            **para:

        Returns:

        """
        for key, value in para.items():
            if key not in dir(self.model):
                print("设置失败")
                continue
            if value <= 0 and key != "random_state":
                continue
            if key == "learning_rate_init" and value > 1:
                print("设置无效，学习率大于1会让模型无法收敛!!")
                continue
            if value >= 0:
                exec("".join(["self.model.", key, "=", str(value)]))


class DatasetsNew:
    def __init__(self, name):
        """
        初始化数据集的类
        :param name:
        """
        self.data_name = name
        self.data = self.load(name)
        self.__train_rate = 0.75
        self.split()

    def load(self, name):
        """
        导入数据集
        :param name:
        :return:
        """
        if name == data_name['鸢尾花']:
            from sklearn.datasets import load_iris
            data = load_iris()
        elif name == data_name['手写数字']:
            from sklearn.datasets import load_digits
            data = load_digits()
        elif name == data_name['红酒']:
            from sklearn.datasets import load_wine
            data = load_wine()
        elif name == data_name['威斯康辛州乳腺癌']:
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
        else:
            raise ValueError("没有这个数据集")
        return data

    def split(self):
        """
        划分数据集
        :return:
        """
        X, Y = self.data.data, self.data.target
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, train_size=self.__train_rate)

    def setSplitNum(self, train_rate):
        """
        设置训练集的比例
        :param train_rate:
        :return:
        """
        self.__train_rate = train_rate
        self.split()
