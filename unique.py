import websocket, json, threading, os, pyaudio, subprocess, time, re
from sklearn.preprocessing import normalize
from tkinter import Text, Frame, Tk, Button, END, messagebox, filedialog, INSERT, DISABLED
from PIL import ImageTk, Image
from wordcloud import WordCloud, ImageColorGenerator
import jieba, sys
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import numpy as np
from matplotlib.animation import FuncAnimation
plt.rcParams['font.sans-serif'] = ['SimHei']

# 用来转换数字
class Number_Convert():
    def __init__(self):

        self.number_map = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8,
                           '九': 9}  # 1-9数字
        self.unit_map = {'十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}  # 数字单位

    def __operate(self, num_str: str):  # 这个和下面呢个____operate1都是处理字符串的函数，别调用
        Num = 0
        unit = False
        for index, i in enumerate(num_str[::-1]):
            if i in self.number_map.keys():
                Num += self.number_map[i] * (unit if unit else 1)
            elif index != len(num_str) - 1:
                unit = self.unit_map[i]
            else:
                Num += self.unit_map[i]
        return Num

    def __operate1(self, strings: str):  # 处理字符串的，分成了三种情况，有“亿”，无“亿”有“万”， 无“亿”无“万”

        if '亿' in strings:
            strings2 = strings.split('亿')
            Num1 = 0
            for index0, i in enumerate(strings2):
                Num = 0
                if len(i.split('万')) != 1:
                    for index, j in enumerate(i.split('万')):
                        Num += self.__operate(j) * (10000 if index == 0 else 1)
                else:
                    Num += self.__operate(i.split('万')[0])

                Num1 += Num * (self.unit_map['亿'] if index0 == 0 else 1)
            return Num1
        elif '万' in strings:
            Num = 0
            for index, j in enumerate(strings.split('万')):
                Num += self.__operate(j) * (self.unit_map['万'] if index == 0 else 1)
            return Num
        else:
            return self.__operate(strings)

    def num_convert3(self, test_strings: str):
        self.NumList = []  # 装数字的列表
        self.converted_strings = ''  # 转化后的字符串
        self.test_strings = test_strings.replace("什", "【·&……】")
        for index0, Str in enumerate(self.test_strings):  # 遍历一下字符串
            try:  # 如果已经遍历完一串数字，那我们要把当前的位置移动到这一串数字之后，再继续遍历下面的内容
                if index0 < index1:
                    continue
            except:  # 如果报错了，表示还没有遍历过任何一串数字，所以j还没有定义
                pass
            if (Str.isnumeric() and not Str.isdigit()) or (
                    Str == '两'):  # 如果遍历到的那个字符是中文数字，不是阿拉伯数字，那就从那个字符开始，遍历那个字符以及之后的字符串部分
                for index1, Str2 in enumerate(self.test_strings[index0:]):
                    if (not Str2.isnumeric()) and Str2 != '两':  # 如果遍历到不是数字的字符，表示这一串数字遍历完了，那就开始将这一串中文数字转化成阿拉伯数字
                        Num = self.__operate1(self.test_strings[index0:index0 + index1])
                        self.converted_strings += str(Num)
                        self.NumList.append((Num, index0))
                        index1 = index0 + index1  # 让index1表示当前遍历到的字符串的位置
                        break
            else:
                self.converted_strings += Str  # 把当前遍历到的内容给Str
        for num in re.compile('\d+').finditer(self.test_strings):
            self.NumList.append((int(num.group()), num.span()[0]))
        return [self.converted_strings.replace("【·&……】", "什"), sorted(self.NumList, key=lambda x: x[1])]


######################本地化语音########################

class Yuyin_local():
    def __init__(self, record_time_s, local_yuyinPath, asyn=True, filename=''):
        os.system("chcp 65001")  # 切换语音为Unicode (UTF-8)
        vbsPath = os.path.join(local_yuyinPath, "runbat.vbs")
        subprocess.call(f"cscript  {vbsPath}", stdout=None, stdin=None)
        self.ws_app = websocket.WebSocketApp("ws://127.0.0.1:10086",
                                             on_open=lambda ws: self.on_open(ws, record_time_s),  # 连接建立后的回调
                                             on_message=self.on_message,  # 接收消息的回调
                                             on_error=self.on_error,
                                             on_close=self.on_close,
                                             on_data=self.on_data,
                                             on_ping=self.on_ping,
                                             on_pong=self.on_pong,
                                             on_cont_message=self.on_cont_message
                                             )

        self.total_sentance = ''  # 存放语音识别的内容
        self.asyn = asyn
        self.filename = filename
        p = pyaudio.PyAudio()
        self.stream = p.open(format=pyaudio.paInt16,
                             channels=1,
                             rate=16000,
                             input=True,
                             frames_per_buffer=1024,
                             )

    def run(self):
        self.ws_app.run_forever()

    def on_message(self, ws, message):
        """
        接收服务端返回的消息
        :param ws:
        :param message: json格式，自行解析
        :return:
        """
        self.Dict = json.loads(message)
        # 如果判断一段话结束了，就把这段话存储到self.total_sentance里面，下一段话就可以在这一段话之后继续拼接
        if self.Dict['type'] == "final_result" and self.asyn:
            self.total_sentance += json.loads(self.Dict['nbest'])[0]['sentence']
        # server_ready是我们发送开始帧的时候传回来的数据，我们不需要读取
        if self.Dict['type'] != "server_ready" and self.asyn:
            print('\r', self.total_sentance + json.loads(self.Dict['nbest'])[0]['sentence'], end='', flush=True)

    def on_open(self, ws, record_time_s):
        """
        连接后发送数据帧
        :param  websocket.WebSocket ws:
        :return:
        """

        def run(*args):
            """
            主程序
            :param args:
            :return:
            """
            '''
             发送二进制音频数据，注意每个帧之间需要有间隔时间
             :param ws:
             :param record_time_s: 录音时长，单位是秒
             :return:
             '''
            # 开始参数帧,写死的，不要动
            startData = '{"signal":"start","nbest":1,"continuous_decoding":true}'
            ws.send(startData, websocket.ABNF.OPCODE_TEXT)
            if self.asyn:  # 一边说一边识别
                print(f'开始录音，持续{record_time_s}秒')
                for i in range(0, int(16000 / 1024) * record_time_s):
                    data = self.stream.read(1024)
                    ws.send(data, websocket.ABNF.OPCODE_BINARY)
                print('\n录音结束')
            else:
                chunk_ms = 160  # 160ms的录音
                chunk_len = int(16000 * 2 / 1000 * chunk_ms)
                with open(self.filename, 'rb') as f:
                    pcm = f.read()
                index = 0
                total = len(pcm)
                print("开始识别")
                total_time_s = time.time() + total / 32000
                while index < total:
                    end = index + chunk_len
                    if end >= total:
                        # 最后一个音频数据帧
                        end = total
                    body = pcm[index:end]
                    ws.send(body, websocket.ABNF.OPCODE_BINARY)
                    index = end
                    time.sleep(chunk_ms / 1000.0)  # ws.send 也有点耗时，这里没有计算
                    last_time = round(total_time_s - time.time(), 1)
                    print('\r', "识别中，预计还差", last_time if last_time > 0 else 0.00, "秒", end='', flush=True)

            # 避免时间过短导致句子还没结束，函数就结束了
            if self.Dict:
                if self.Dict["type"] == "partial_result":
                    self.total_sentance += json.loads(self.Dict['nbest'])[0]['sentence']
            # 发送结束帧，写死的，不要动
            endData = '{"signal": "end"}'
            ws.send(endData, websocket.ABNF.OPCODE_TEXT)
            self.ws_app.close()

        threading.Thread(target=run).start()

    def on_error(self, ws, error, c, d):
        print("\n出现了错误")
        # print(error,c,d)

    def on_close(self, we, c, d):
        print("\n识别结束")

    def on_data(self, we, message, message_len, isSend):
        pass

    def on_ping(self):
        pass

    def on_pong(self):
        pass

    def on_cont_message(self):
        pass


######################bp神经网络###################
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

######################词云###################


system_platform = sys.platform
# 设置路径
if 'win' in system_platform:
    # 获取当前文件的位置
    file_path = os.path.join(os.getcwd().split('blockly-electron')[0], 'blockly-electron')
    if not os.path.exists(file_path):
        if os.path.exists(os.path.join(os.getcwd(), "resources")):
            file_path = os.getcwd()
    main_path = os.path.join(file_path, 'resources', 'assets', 'class').replace("\\", "/")


def generateCloud():
    root = Tk()
    root.geometry("784x400")
    root.resizable(0, 0)
    root.title("词云-左侧输入文章或导入txt文件，右侧输出词云图片")
    app = Application(root)
    root.mainloop()


class Application(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.creatWidget()
        self.img = None
        self.fname_stop = "".join([main_path, "/txt", '/hit_stopwords.txt']).replace("\\", "/").replace(
            os.getcwd().replace("\\", "/"), ".")
        self.fname_mask = "".join([main_path, "/picture", '/owl.jpeg']).replace("\\", "/").replace(
            os.getcwd().replace("\\", "/"), ".")
        self.fname_font = "".join([main_path, "/fonts", '/SourceHanSerifK-Light.otf']).replace("\\", "/").replace(
            os.getcwd().replace("\\", "/"), ".")

    def creatWidget(self):
        self.w1 = Text(self, width=50, heigh=30)  # 宽度为80个字母(40个汉字)，高度为1个行高
        self.w2 = Text(self, width=50, heigh=30, bg='white')
        self.w2.configure(state=DISABLED)
        self.w1.pack(side="left")
        self.w2.pack(side="left")
        self.button1 = Button(self, text="开始转换", command=self.convert)
        self.button1.pack()
        self.buttom2 = Button(self, text="读取txt文件", command=self.seletFile)
        self.buttom2.pack()
        self.buttom3 = Button(self, text="清空内容", command=self.clear)
        self.buttom3.pack()
        self.buttom4 = Button(self, text="保存词云图片", command=self.saveImg)
        self.buttom4.pack()

    # 返回信息
    def convert(self):
        allText = self.w1.get(1.0, END)
        if len(allText.split()) == 0:
            messagebox.showinfo("错误", "内容不能为空")
        else:
            print("正在转换...")
            wcd = self.generate_wordCloud(allText)
            self.img = wcd.to_image()
            self.photo = ImageTk.PhotoImage(self.img)
            self.w2.delete(1.0, END)
            self.w2.image_create(1.0, image=self.photo)
            print("转换成功")

    def count_frequencies(self, word_list):
        freq = dict()
        for w in word_list:
            if w not in freq.keys():
                freq[w] = 1
            else:
                freq[w] += 1
        return freq

    def plt_imshow(self, x, ax=None, show=True):
        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(x)
        ax.axis("off")
        if show: plt.show()
        return ax

    def generate_wordCloud(self, text):
        # 读取需要过滤的词
        with open(self.fname_stop, encoding='utf8') as f:
            STOPWORDS_CH = f.read().split()
        # 处理输入的文章，把一些没用的词过滤掉
        word_list = []
        for w in jieba.cut(text):
            if w not in STOPWORDS_CH and len(w) > 1:
                word_list.append(w)
        # 返回字典，字典里面的值是其键对应词汇出现的频率
        freq = self.count_frequencies(word_list)
        # 处理图片
        im_mask = np.array(Image.open(self.fname_mask))
        im_colors = ImageColorGenerator(im_mask)
        # 生成词云
        wcd = WordCloud(font_path=self.fname_font,  # 中文字体
                        background_color='white',
                        mode="RGBA",
                        mask=im_mask,
                        )
        wcd.generate_from_frequencies(freq)
        wcd.recolor(color_func=im_colors)

        return wcd

    def seletFile(self):
        path = filedialog.askopenfilename()
        if path.endswith(".txt"):
            with open(path, 'r', encoding='utf-8') as f:
                self.w1.insert(INSERT, f.read())
        else:
            messagebox.showinfo("错误", "请选择txt文件")

    def clear(self):
        self.w1.delete(1.0, END)

    def saveImg(self):
        try:
            file_path = filedialog.asksaveasfilename(title=u'保存文件')
            print(file_path + ".png")
            if file_path.endswith(".png"):
                self.img.save(file_path)
            else:
                print(file_path + ".png")
                self.img.save(file_path + ".png")
            messagebox.showinfo("成功", "保存成功")
        except:
            messagebox.showinfo("错误", "保存失败")


######################以下是playsound修改版本###################
import logging

logger = logging.getLogger(__name__)


class PlaysoundException(Exception):
    pass


def _canonicalizePath(path):
    """
    Support passing in a pathlib.Path-like object by converting to str.
    """
    import sys
    if sys.version_info[0] >= 3:
        return str(path)
    else:
        # On earlier Python versions, str is a byte string, so attempting to
        # convert a unicode string to str will fail. Leave it alone in this case.
        return path


def _playsoundWin(sound, block=True):
    '''
    Utilizes windll.winmm. Tested and known to work with MP3 and WAVE on
    Windows 7 with Python 2.7. Probably works with more file formats.
    Probably works on Windows XP thru Windows 10. Probably works with all
    versions of Python.

    Inspired by (but not copied from) Michael Gundlach <gundlach@gmail.com>'s mp3play:
    https://github.com/michaelgundlach/mp3play

    I never would have tried using windll.winmm without seeing his code.
    '''
    sound = _canonicalizePath(sound)

    if any((c in sound for c in ' "\'()')):
        from os import close, remove
        from os.path import splitext
        from shutil import copy
        from tempfile import mkstemp

        fd, tempPath = mkstemp(prefix='PS',
                               suffix=splitext(sound)[1])  # Avoid generating files longer than 8.3 characters.
        logger.info(
            'Made a temporary copy of {} at {} - use other filenames with only safe characters to avoid this.'.format(
                sound, tempPath))
        copy(sound, tempPath)
        close(fd)  # mkstemp opens the file, but it must be closed before MCI can open it.
        try:
            _playsoundWin(tempPath, block)
        finally:
            remove(tempPath)
        return

    from ctypes import c_buffer, windll
    from time import sleep

    def winCommand(*command):
        bufLen = 600
        buf = c_buffer(bufLen)
        command = ' '.join(command)  # .encode('utf-16')
        errorCode = int(
            windll.winmm.mciSendStringW(command, buf, bufLen - 1, 0))  # use widestring version of the function
        if errorCode:
            errorBuffer = c_buffer(bufLen)
            windll.winmm.mciGetErrorStringW(errorCode, errorBuffer,
                                            bufLen - 1)  # use widestring version of the function
            exceptionMessage = ('\n    Error ' + str(errorCode) + ' for command:'
                                                                  '\n        ' + command.decode('utf-16') +
                                '\n    ' + errorBuffer.raw.decode('utf-16').rstrip('\0'))
            logger.error(exceptionMessage)
            raise PlaysoundException(exceptionMessage)
        return buf.value

    if '\\' in sound:
        sound = '"' + sound + '"'

    try:
        logger.debug('Starting')
        winCommand(u'open {}'.format(sound))
        winCommand(u'play {}{}'.format(sound, ' wait' if block else ''))
        logger.debug('Returning')
    finally:
        try:
            winCommand(u'close {}'.format(sound))
        except PlaysoundException:
            logger.warning(u'Failed to close the file: {}'.format(sound))
            # If it fails, there's nothing more that can be done...
            pass


def _handlePathOSX(sound):
    sound = _canonicalizePath(sound)

    if '://' not in sound:
        if not sound.startswith('/'):
            from os import getcwd
            sound = getcwd() + '/' + sound
        sound = 'file://' + sound

    try:
        # Don't double-encode it.
        sound.encode('ascii')
        return sound.replace(' ', '%20')
    except UnicodeEncodeError:
        try:
            from urllib.parse import quote  # Try the Python 3 import first...
        except ImportError:
            from urllib import quote  # Try using the Python 2 import before giving up entirely...

        parts = sound.split('://', 1)
        return parts[0] + '://' + quote(parts[1].encode('utf-8')).replace(' ', '%20')


def _playsoundOSX(sound, block=True):
    '''
    Utilizes AppKit.NSSound. Tested and known to work with MP3 and WAVE on
    OS X 10.11 with Python 2.7. Probably works with anything QuickTime supports.
    Probably works on OS X 10.5 and newer. Probably works with all versions of
    Python.

    Inspired by (but not copied from) Aaron's Stack Overflow answer here:
    http://stackoverflow.com/a/34568298/901641

    I never would have tried using AppKit.NSSound without seeing his code.
    '''
    try:
        from AppKit import NSSound
    except ImportError:
        logger.warning("playsound could not find a copy of AppKit - falling back to using macOS's system copy.")
        sys.path.append('/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/PyObjC')
        from AppKit import NSSound

    from Foundation import NSURL
    from time import sleep

    sound = _handlePathOSX(sound)
    url = NSURL.URLWithString_(sound)
    if not url:
        raise PlaysoundException('Cannot find a sound with filename: ' + sound)

    for i in range(5):
        nssound = NSSound.alloc().initWithContentsOfURL_byReference_(url, True)
        if nssound:
            break
        else:
            logger.debug('Failed to load sound, although url was good... ' + sound)
    else:
        raise PlaysoundException('Could not load sound with filename, although URL was good... ' + sound)
    nssound.play()

    if block:
        sleep(nssound.duration())


def _playsoundNix(sound, block=True):
    """Play a sound using GStreamer.

    Inspired by this:
    https://gstreamer.freedesktop.org/documentation/tutorials/playback/playbin-usage.html
    """
    sound = _canonicalizePath(sound)

    # pathname2url escapes non-URL-safe characters
    from os.path import abspath, exists
    try:
        from urllib.request import pathname2url
    except ImportError:
        # python 2
        from urllib import pathname2url

    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst

    Gst.init(None)

    playbin = Gst.ElementFactory.make('playbin', 'playbin')
    if sound.startswith(('http://', 'https://')):
        playbin.props.uri = sound
    else:
        path = abspath(sound)
        if not exists(path):
            raise PlaysoundException(u'File not found: {}'.format(path))
        playbin.props.uri = 'file://' + pathname2url(path)

    set_result = playbin.set_state(Gst.State.PLAYING)
    if set_result != Gst.StateChangeReturn.ASYNC:
        raise PlaysoundException(
            "playbin.set_state returned " + repr(set_result))

    # FIXME: use some other bus method than poll() with block=False
    # https://lazka.github.io/pgi-docs/#Gst-1.0/classes/Bus.html
    logger.debug('Starting play')
    if block:
        bus = playbin.get_bus()
        try:
            bus.poll(Gst.MessageType.EOS, Gst.CLOCK_TIME_NONE)
        finally:
            playbin.set_state(Gst.State.NULL)

    logger.debug('Finishing play')


def _playsoundAnotherPython(otherPython, sound, block=True, macOS=False):
    '''
    Mostly written so that when this is run on python3 on macOS, it can invoke
    python2 on macOS... but maybe this idea could be useful on linux, too.
    '''
    from inspect import getsourcefile
    from os.path import abspath, exists
    from subprocess import check_call
    from threading import Thread

    sound = _canonicalizePath(sound)

    class PropogatingThread(Thread):
        def run(self):
            self.exc = None
            try:
                self.ret = self._target(*self._args, **self._kwargs)
            except BaseException as e:
                self.exc = e

        def join(self, timeout=None):
            super().join(timeout)
            if self.exc:
                raise self.exc
            return self.ret

    # Check if the file exists...
    if not exists(abspath(sound)):
        raise PlaysoundException('Cannot find a sound with filename: ' + sound)

    playsoundPath = abspath(getsourcefile(lambda: 0))
    t = PropogatingThread(
        target=lambda: check_call([otherPython, playsoundPath, _handlePathOSX(sound) if macOS else sound]))
    t.start()
    if block:
        t.join()


from platform import system

system = system()

if system == 'Windows':
    playsound = _playsoundWin
elif system == 'Darwin':
    playsound = _playsoundOSX
    import sys

    if sys.version_info[0] > 2:
        try:
            from AppKit import NSSound
        except ImportError:
            logger.warning(
                "playsound is relying on a python 2 subprocess. Please use `pip3 install PyObjC` if you want playsound to run more efficiently.")
            playsound = lambda sound, block=True: _playsoundAnotherPython(
                '/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python', sound, block, macOS=True)
else:
    playsound = _playsoundNix
    if __name__ != '__main__':  # Ensure we don't infinitely recurse trying to get another python instance.
        try:
            import gi

            gi.require_version('Gst', '1.0')
            from gi.repository import Gst
        except:
            logger.warning(
                "playsound is relying on another python subprocess. Please use `pip install pygobject` if you want playsound to run more efficiently.")
            playsound = lambda sound, block=True: _playsoundAnotherPython('/usr/bin/python3', sound, block, macOS=False)

del system
