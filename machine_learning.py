import os,sys

import numpy as np
from sklearn import datasets as DT #这个别删
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from typing import Any
system_platform = sys.platform
main_path = '/home/pi/class/'  # 读取和保存文件所用主文件夹
if 'win' in system_platform:
    # 获取当前文件的位置
    file_path = os.path.join(os.getcwd().split('blockly-electron')[0], 'blockly-electron')
    if not os.path.exists(file_path):
        if os.path.exists(os.path.join(os.getcwd(), "resources")):
            file_path = os.getcwd()
    main_path = os.path.join(file_path, 'resources', 'assets', 'class').replace("\\", "/")
picture_path = os.path.join(main_path , 'picture/')  # 图片文件夹
model_path = os.path.join(main_path , 'model/')  # 识别模型文件夹
#查找文件夹下存在的pkl模型的名字（没有pkl后缀）
def findExisModel():
    '''
    查找文件夹下存在的pkl模型的名字（没有pkl后缀）
    Returns:

    '''
    print(model_path)
    if os.path.exists(model_path):
        return [model_name[:-4] for model_name in os.listdir(model_path) if
                                  model_name.endswith(".pkl")]  # model文件夹下面存在的pkl模型文件名的列表
    else:
        return []

class Datasets():
    '''
    初始化数据集的类，用于获取各种数据集
    可选数据集：
                "digits": "手写数字"
                "iris": "鸢尾花"
                "breast_cancer": "乳腺癌"
                "diabetes": "糖尿病"
                "boston": "房价"
    '''
    def __init__(self, datasets_name:str):
        self.datasets_name = datasets_name
        self.__get_datasets()

    def __call__(self):
        return self.dataset

    def __get_datasets(self):
        '''
        加载数据集(作为私密成员，防止被多次调用)
        Returns:

        '''
        exec("self.dataset=DT.load_"+self.datasets_name+"()")
        self.feature = self.dataset.data
        self.label = self.dataset.target

    def split(self,test_size:int=30):
        self.feature_train,  self.feature_test,  self.label_train,  self.label_test = train_test_split(self.feature, self.label, test_size=test_size/100)
        return (self.feature_train,  self.label_train),(self.feature_test,  self.label_test)



def feature_label(dataset:Any):
    '''
    划分数据集
    Args:
        dataset:将要划分的数据集

    Returns: 划分之后的数据集

    '''
    if callable(dataset): #传入了一个类对象
        return dataset().data, dataset().target
    elif 'data' in dir(dataset): #传入了一个基本的数据集，没有进行数据划分
        return dataset.data, dataset.target
    else: #进行了数据集划分
        return dataset[0], dataset[1]

class Model():
    '''
    初始化训练和识别数据集的模型
    支持的模型有：
    决策树
    随机森林
    k近邻
    逻辑回归
    支持向量机
    神经网络
    '''
    def __init__(self, model_name:str='', myModel_name:str=''):
        self.model_name = model_name #模型名字
        self.myModel_name = myModel_name #自己加载的模型的名字
        self.test_score = '你还尚未进行模型的验证，可以先用验证集对模型进行验证，在打印这个块噢~'
        self.isTrain = False if model_name else True #模型是否已经训练,如果是加载保存的模型，那肯定是已经训练了
        self.pred = '你要先预测，才会有结果噢~'
        self.model_name_dict = { #可选的机器学习算法模型
            "Tree": ("决策树", "tree", "DecisionTreeClassifier"),
            "RandomForest": ("随机森林", "ensemble", "RandomForestClassifier"),
            "KNeighbors": ("k近邻", "neighbors", "KNeighborsClassifier"),
            "LogisticRegress": ("逻辑回归", "linear_model", "LogisticRegression"),
            "SVM": ("支持向量机", "svm", "SVC"),
            "MLPClassifier":("神经网络", "neural_network", "MLPClassifier")
        }
        from sklearn.neural_network import  MLPClassifier
        self.myModel_name_list = findExisModel() #查找class文件夹下面存在的模型的名字
        self.classifier = self.__load_model()  # 调用模型

    def __str__(self):
        return self.model_name_dict[self.model_name][0]+"模型" if self.model_name else self.myModel_name+"模型"

    def __load_model(self):
        '''
        加载模型
        Returns:

        '''
        if not(self.model_name or self.myModel_name):
            raise "你没有输入你的模型名称"
        if self.model_name:
            exec("".join(["from sklearn.", self.model_name_dict[self.model_name][1],
                          " import ",self.model_name_dict[self.model_name][2]]))
            return eval(self.model_name_dict[self.model_name][2])()
        if (self.myModel_name[:-4] in self.myModel_name_list) or os.path.isabs(self.myModel_name):
            return joblib.load((model_path if os.path.isabs(self.myModel_name)==False else "")+ \
                                        self.myModel_name+ (".pkl" if self.myModel_name.split(".")[-1] != "pkl" else ""))
        else:
            exis_model = "".join(["-"*10,"\n","\n".join(self.myModel_name_list),"\n","-"*10])
            err = f"\n没有找到名字为'{self.myModel_name}'的模型，文件夹存在的模型文件只有这些：\n{exis_model}"
            raise FileNotFoundError(err)

    def train(self, train_datasets:np.ndarray):
        '''
        训练模型
        Args:
            train_datasets:  用于训练的数据集

        Returns:

        '''
        feature, label = feature_label(train_datasets)
        print("**开始训练**")
        self.classifier.fit(feature, label)
        print("训练完成！\n")
        self.isTrain = True

    def test(self, test_datasetsl:np.ndarray):
        '''
        验证模型的好坏
        Args:
            test_datasetsl: 用于验证的验证集

        Returns:

        '''
        if not self.isTrain:
            print("你的模型还没进行训练！")
        else:
            print("**开始用验证集来验证模型**")
            feature, label = feature_label(test_datasetsl)
            print("验证完成！\n")
            self.test_score = "".join(["准确率为：", str(round(accuracy_score(self.classifier.predict(feature), label)*100, 2))," %"])

    def predict(self, feature:np.ndarray):
        '''
        将模型用于预测，这个模型一般就是已经训练完且已经验证完毕的模型
        Args:
            feature: 输入的特征

        Returns: 输出预测的结果

        '''
        if not self.isTrain:
            print("你的模型还没进行训练！")
        else:
            self.pred = self.classifier.predict(feature)

    def save(self, name:str):
        '''
        保存模型
        Args:
            name: 保存的模型的名字，需要以pkl为后缀

        Returns:

        '''
        if name.split(".")[-1] != "pkl":
            raise NameError("名字格式错误！保存的名字必须有个.pkl后缀比如myFirstModel.pkl")
        try:
            joblib.dump(self.classifier, (model_path if os.path.isabs(name)==False else "")+ name) #模型后缀名统一为.pkl
            print("保存模型成功！")
        except:
            print('保存模型失败！')

    def setParameter(self, **para:Any):
        '''
        为各种模型设置参数
        Args:
            **para:

        Returns:

        '''
        for key,value in para.items():
            if value<=0 and key != "random_state":
                continue
            if value >=0:
                exec("".join(["self.classifier.",key, "=", str(value)]))
