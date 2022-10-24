# 作者：tomoya
# 创建：2022-09-29
# 更新：2022-09-29
# 用意：词云
import os
from tkinter import Text, Frame, Tk, Button, END, messagebox, filedialog, INSERT, DISABLED
from PIL import ImageTk, Image
from wordcloud import WordCloud, ImageColorGenerator
import jieba, sys
import matplotlib.pyplot as plt
import numpy as np

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
        try:
            wcd.generate_from_frequencies(freq)
        except ValueError:
            raise ValueError("词云官方bug!!输入的这些词没办法构成词云")
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
