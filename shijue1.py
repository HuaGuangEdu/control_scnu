import cv2
import sys
from PIL import Image
import numpy as np
from collections import deque
import math
from .shijue0 import basicImg
from .shijue2 import AdvancedImg
import os
from typing import Any
from .util.opencv_tool import cv2AddChineseText, draw_dotted_rect
import shutil
from .util.all_path import picture_path, model_path, class_path, system_platform

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if 'win' in system_platform:
    import mediapipe as mp
else:
    try:
        import mediapipe as mp
    except (ImportError, ModuleNotFoundError):
        print('你的树莓派没有安装mediapipe。现在安装……')
        os.system('sudo pip3 install mediapipe-rpi4')

from cvzone.SelfiSegmentationModule import SelfiSegmentation

camera_pos_path = os.path.join(class_path, 'camera_pos')
face_recognize_path = os.path.join(class_path, 'data/face_recognize')
color_cluster_path = os.path.join(class_path, 'data/color_cluster')

items_num = {0: '0', 1: '1', 2: '2', 3: '3',
             4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}  # 方向数字指示牌
items_dir = {9: '左转', 10: '右转'}


def new_file(path: str):
    """
    创建文件夹
    Args:
        path:

    Returns:

    """
    if os.path.isdir(path):
        pass
    else:
        print('未找到该文件夹。开始创建文件夹')
        os.makedirs(path)


def zeros_like(img: np.ndarray):
    """
    创建一个与img图像等尺寸的全零图像
    Args:
        img: 输入图像

    Returns:输出全零图像

    """
    return np.zeros_like(img)


def line(img: np.ndarray, point1: tuple, point2: tuple):
    """
    在图片的某两点之间画线
    Args:
        img: 
        point1: 
        point2: 

    Returns:

    """
    cv2.line(img, point1, point2, (255, 0, 0))


class Img(basicImg, AdvancedImg):
    """
    新的视觉类，用来实现高级操作
    """

    def __init__(self):
        super(Img, self).__init__()
        AdvancedImg.__init__(self)
        """
        人脸识别属性
        """
        self.ID = ''
        self.model_ID = self.ID
        self.data_path = None
        self.save_model_name = None
        self.face_model = model_path + 'face.xml'
        self.face_detector = cv2.CascadeClassifier(self.face_model)
        self.faces = None
        self.ids = None
        self.face_name = 'none'
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.Type = None
        """
        人脸检测属性
        """
        self.path = ''
        self.cascade = None
        """
        戴帽子、口罩属性
        """
        self.pht = ''
        self.flag_cap = 0
        self.flag_mask = 0
        """
        形状检测属性
        """
        self.polygon_corners = 0
        self.shape_direction_s = 'None'
        self.shape_direction_p = 'None'
        self.cx = 0
        self.cy = 0
        self.radium = 0
        self.circle_center = [0, 0]
        self.area = 0
        self.length = 0
        self.shape_type = 'None'
        self.draw_or_not = 0
        """
               mediapipe属性
         """
        self.fingertip = {}
        self.body_menu = {}
        # self.mp_drawing = mp.solutions.drawing_utils
        # self.mp_drawing_styles = mp.solutions.drawing_styles

    """
            以下 change_ID、get_data、get_info、train、predict_init、predict函数都是用来进行人脸识别的
            change_ID: 用来获取想要进行人脸注册的人名
            get_data: 用来获取该人名对应的人脸图片
            get_info: 用来获取get_data得到的数据集中的人脸标签和信息
            train:  根据get_info的信息进行训练
            predict_init:初始化检测器
            predict: 进行预测
            示例1 获取数据集，训练，并进行预测：
            I = Img()
            I.camera(0)
            I.change_ID('JACK')
            I.get_data(50)
            I.train()
            I.predict_init()
            while True:
                I.get_img()
                I.predict()
                I.delay(1)
            示例2 直接读取某一个保存好的模型进行预测：
            I = Img()
            I.camera(0)
            I.predict_init('JACK')
            while True:
                I.get_img()
                I.predict()
                I.delay(1)
        """

    def change_ID(self, ID: str):
        """
        用来获取想要进行人脸注册的人名
        Args:
            ID:

        Returns:

        """
        self.ID = ID
        self.data_path = face_recognize_path + self.ID + '/'
        new_file(self.data_path)
        self.save_model_name = model_path + self.ID + '.yml'

    def get_data(self, pic_num: int = 50, time: int = 1):
        """
        用来获取该人名对应的人脸图片
        Args:
            pic_num:
            time:

        Returns:

        """
        i = 0
        while True:
            if i >= pic_num:
                cv2.destroyAllWindows()
                break
            self.get_img()
            self.name_windows('img')
            self.show_image('img')
            pic_my_name = self.data_path + self.ID + '_' + str(i) + '.jpg'
            i += 1
            cv2.imwrite(pic_my_name, self.img)
            self.delay(time)

    def get_info(self):
        """
        用来获取get_data得到的数据集中的人脸标签和信息
        Returns:

        """
        try:
            facesSamples = []
            ids = []
            self.data_ = face_recognize_path
            self.names = []
            imgP = []
            file = os.listdir(self.data_)
            for i in range(len(file)):
                next_file_path = self.data_ + file[i] + '/'
                for f in os.listdir(next_file_path):
                    if f.split('.')[1] != 'jpg':
                        continue
                    self.names.append(file[i])
                    imgP.append(os.path.join(next_file_path, f))
            # print(self.names)
            # print(len(imgP), len(self.names))
            for im in range(len(imgP)):
                # 打开图片,黑白化
                PIL_img = Image.open(imgP[im]).convert('L')
                # 将图像转换为数组，以黑白深浅
                # PIL_img = cv2.resize(PIL_img, dsize=(400, 400))
                img_numpy = np.array(PIL_img, 'uint8')
                # print(img_numpy)
                # 获取图片人脸特征
                faces = self.face_detector.detectMultiScale(img_numpy)
                # 获取每张图片的id和姓名
                # 预防无面容照片
                for x, y, w, h in faces:
                    ids.append(im)
                    facesSamples.append(img_numpy[y:y + h, x:x + w])
            self.faces = facesSamples
            self.ids = ids
            print('成功读取标签')
            return True
        except FileNotFoundError:
            print('请检查数据集是否在路径内')
            return False

    def train(self):
        """训练人脸识别模型"""
        self.get_info()
        self.recognizer.train(self.faces, np.array(self.ids))
        self.recognizer.write(self.save_model_name)
        print('训练完毕，模型已保存到：', self.save_model_name, '模型名称为：', self.ID)

    def predict_init(self, m_name: str = ''):
        """
        初始化人脸识别器
        Args:
            m_name:

        Returns:

        """
        ret = self.get_info()
        if ret == False:
            return False
        model_name = self.save_model_name
        if len(m_name) != 0:
            model_name = model_path + m_name + '.yml'
        # print(model_name, type(model_name))
        if os.path.isfile(model_name):
            self.recognizer.read(model_name)
        # except:
        #     print('please get your model')

    def predict(self):
        """
        进行人脸识别
        Returns:

        """
        try:
            img = self.img
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度
            face = self.face_detector.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (100, 100), (300, 300))
            for x, y, w, h in face:
                cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
                cv2.circle(img, center=(x + w // 2, y + h // 2), radius=w // 2, color=(0, 255, 0), thickness=1)
                # 人脸识别
                ids, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
                print(len(self.names), ids)
                print('名字', str(self.names[ids - 1]), '置信值：', confidence)
                try:
                    if confidence > 80:
                        cv2.putText(img, 'unknown', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
                    else:
                        cv2.putText(img, str(self.names[ids - 1]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                    (0, 255, 0), 1)
                        self.face_name = str(self.names[ids - 1])
                except:
                    pass
            self.name_windows('face recognize result')
            self.show_image('face recognize result', img)
        except cv2.error:
            print('请到官网进行反馈')

    """
        以下 face_detect_init、face_detect、face_cap、face_mask 是用来实现人脸检测以及给人脸戴帽子戴口罩的
        face_detect_init: 输入选择用来识别的人脸模型
        face_detect: 进行人脸检测
        face_cap: 调用face_detect，并对框出来的人脸进行贴图
        face_mask：同上
        示例1 人脸检测：
        I = Img()
        I.camera(0)
        I.face_detect_init('face')
        I.name_windows('output')
        while True:
            I.get_img()
            I.face_detect()
            I.show_image('output')
            I.delay(1)
        示例2 戴帽子（戴口罩同理）：
        I = Img()
        I.camera(0)
        I.face_detect_init('face')
        I.name_windows('output')
        while True:
            I.get_img()
            I.face_cap()
            I.show_image('output')
            I.delay(1)
    """

    def face_detect_init(self, model_name: str):
        """
        初始化人脸检测器
        Args:
            model_name:

        Returns:

        """
        self.mpFaceDetection = mp.solutions.face_detection  # 人脸识别
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(0.5)
        self.face_data = '没有人'

    def face_detect(self, cool: bool = False, draw: bool = True):
        """
        开始人脸检测
        Args:
            cool:
            draw:

        Returns:

        """
        img_new = self.img.copy()
        imgRGB = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections == None:
            self.face_data = '没有人'
            return img_new, bboxs
        if self.results.detections:
            self.face_data = f'检测到{len(self.results.detections)}个人'
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img_new.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    x, y, w, h = bbox
                    x1, y1 = x + w, y + h
                    if cool:
                        cv2.rectangle(img_new, bbox, (255, 0, 255), 1)
                        # 下面这一堆只是为了画出来的矩形很酷
                        cv2.line(img_new, (x, y), (x + 30, y), (255, 0, 255), 5)
                        cv2.line(img_new, (x, y), (x, y + 30), (255, 0, 255), 5)
                        cv2.line(img_new, (x1, y), (x1 - 30, y), (255, 0, 255), 5)
                        cv2.line(img_new, (x1, y), (x1, y + 30), (255, 0, 255), 5)
                        cv2.line(img_new, (x, y1), (x + 30, y1), (255, 0, 255), 5)
                        cv2.line(img_new, (x, y1), (x, y1 - 30), (255, 0, 255), 5)
                        cv2.line(img_new, (x1, y1), (x1 - 30, y1), (255, 0, 255), 5)
                        cv2.line(img_new, (x1, y1), (x1, y1 - 30), (255, 0, 255), 5)
                    else:
                        cv2.rectangle(img_new, bbox, (255, 0, 255), 3)
        # self.name_windows('face detect result')
        # self.show_image('face detect result', img_new)
        cv2.imshow('face detect result', img_new)

    def face_cap(self, path: str):
        """
        帽子
        Args:
            path:

        Returns:

        """
        self.pht = os.path.join(camera_pos_path, path + '.jpg')
        self.flag_cap = 1
        self.face_detect()

    def face_mask(self, path: str):
        """
        口罩
        Args:
            path:

        Returns:

        """
        self.pht = os.path.join(camera_pos_path, path + '.jpg')
        self.flag_mask = 1
        self.face_detect()

    """
            以下model_、onnx_detect_new、model_recognize函数是用来调用pt生成的模型的
            model_:默认初始化
            onnx_detect_new: 
            model_recognize: 
            示例：
            I = Img()
            I.camera(0)
            I.model_('lxy1007.proto')
            I.name_windows('img')
            while True:
                I.get_img()
                I.model_recognize()
                result = I.m_data
                I.show_image('img')
                I.delay(1)
        """

    def model_(self, model_name: str = 'lxy1007.proto'):
        """
        加载模型
        Args:
            model_name:

        Returns:

        """
        model_real_path = (model_path if os.path.isabs(model_name) == False else '') + model_name
        self.model = cv2.dnn.readNetFromONNX(model_real_path)  # 如'finally.proto'
        f = model_name.split(".")
        self.item = f[0]
        self.pro = 0
        self.m_data = 'none'

    def onnx_detect_new(self, img: np.ndarray):
        """
        新的onnx模型
        Args:
            img:

        Returns:

        """
        # self.model_()
        img = np.asarray(img, dtype=np.float) / 255
        img = img.transpose(2, 0, 1)
        img = img[np.newaxis, :]
        self.model.setInput(img)
        pro = self.model.forward()
        e_x = np.exp(pro.squeeze() - np.max(pro.squeeze()))
        self.pro = e_x / e_x.sum()

    def model_recognize(self):  # create by realsix on 20211205
        """
        使用初始化的模型进行检测和识别
        Returns:

        """
        mean = np.array([0.485, 0.456, 0.406]) * -1
        std = np.array([0.229, 0.224, 0.225])
        mean = mean[:, np.newaxis, np.newaxis]
        std = std[:, np.newaxis, np.newaxis]
        im3 = np.asarray(
            Image.fromarray(np.uint8(cv2.cvtColor(cv2.resize(self.img_new, (112, 112)), cv2.COLOR_BGR2RGB))),
            dtype=float) / 255
        im3 = im3.transpose(2, 0, 1)
        im3 = np.add(im3, mean)
        im3 = np.divide(im3, std)
        img = np.expand_dims(im3, axis=0)
        self.model.setInput(img)
        out = self.model.forward()
        e_x = np.exp(out.squeeze() - np.max(out.squeeze()))
        self.pro = e_x / e_x.sum()
        if np.max(self.pro) > 0.5:
            classNum = np.argmax(self.pro)
            classNum = int(classNum)
            if classNum in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                self.m_data = items_num[classNum]
        else:
            self.m_data = 'none'

    def number_classify(self):  # create by realsix on 20211205, update on 20220311, only for numbers
        """
        数字识别
        Returns:

        """
        # mean = np.array([0.485, 0.456, 0.406]) * -1
        # std = np.array([0.229, 0.224, 0.225])
        # mean = mean[:, np.newaxis, np.newaxis]
        # std = std[:, np.newaxis, np.newaxis]
        self.color_detect_init('red')
        self.setcolorvalue('red', [173, 43, 46], [180, 255, 255])
        self.color_detect()
        if self.img_new.all() != self.img.all():
            gray = cv2.cvtColor(self.img_new, cv2.COLOR_BGR2GRAY)
            ret, img_new = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imshow('imgnew', img_new)
            cv2.waitKey(1)
            im3 = np.asarray(
                Image.fromarray(np.uint8(cv2.cvtColor(cv2.resize(img_new, (112, 112)), cv2.COLOR_BGR2RGB))),
                dtype=float) / 255
            im3 = im3.transpose(2, 0, 1)
            # im3 = np.add(im3, mean)
            # im3 = np.divide(im3, std)
            img = np.expand_dims(im3, axis=0)
            self.model.setInput(img)
            out = self.model.forward()
            e_x = np.exp(out.squeeze() - np.max(out.squeeze()))
            self.pro = e_x / e_x.sum()
            if np.max(self.pro) > 0.5:
                classNum = np.argmax(self.pro)
                classNum = int(classNum)
                if classNum in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
                    self.m_data = items_num[classNum]
            else:
                self.m_data = 'none'
        else:
            self.m_data = 'none'

    """
           以下 color_detect_init、set_hsv、getpos、setcolorvalue、color_detect是用来进行颜色检测的
        """

    def color_detect_init(self, color: str):
        """
        颜色检测
        Args:
            color:

        Returns:

        """
        self.color_data = 'none'  # 保存颜色的检测结果
        if color == 'red':
            self.color_list_lower = [0, 142, 104]  # 这是红色的数值
            self.color_list_upper = [10, 255, 255]
        elif color == 'green':
            self.color_list_lower = [24, 43, 46]
            self.color_list_upper = [64, 255, 255]
        elif color == 'yellow':
            self.color_list_lower = [26, 43, 46]
            self.color_list_upper = [34, 255, 255]
        elif color == 'blue':
            self.color_list_lower = [80, 43, 46]
            self.color_list_upper = [124, 255, 255]
        elif color == 'orange':
            self.color_list_lower = [11, 43, 46]
            self.color_list_upper = [25, 255, 255]
        elif color == 'black':
            self.color_list_lower = [0, 0, 0]
            self.color_list_upper = [180, 255, 46]
        elif color == 'white':
            self.color_list_lower = [0, 0, 221]
            self.color_list_upper = [180, 30, 255]
        elif color == 'gray':
            self.color_list_lower = [0, 0, 46]
            self.color_list_upper = [180, 43, 220]
        elif color == 'purple':
            self.color_list_lower = [125, 43, 46]
            self.color_list_upper = [155, 255, 255]
        elif color == 'qing':
            self.color_list_lower = [78, 43, 46]
            self.color_list_upper = [99, 255, 255]
        self.colorLower = np.array(self.color_list_lower)  # 这是红色的数值
        self.colorUpper = np.array(self.color_list_upper)
        self.color = color
        # 初始化追踪点的列表
        self.mybuffer = 16
        self.pts = deque(maxlen=self.mybuffer)
        self.counter = 0
        self.Hmax = self.Smax = self.Vmax = 0
        self.Hmin = self.Smin = self.Vmin = 255

    def set_hsv(self, color: str):
        """
        把图片转化成HSV空间
        Args:
            color:

        Returns:

        """
        image = self.img.copy()
        self.HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cv2.imshow("imageHSV", self.HSV)
        cv2.imshow('image', image)
        cv2.setMouseCallback("imageHSV", self.getpos)
        cv2.waitKey(0)
        self.color_list_lower = [self.Hmax, self.Smax, self.Vmax]
        self.color_list_upper = [self.Hmin, self.Smin, self.Vmin]
        self.colorLower = np.array(self.color_list_lower)  # 这是红色的数值
        self.colorUpper = np.array(self.color_list_upper)
        self.color = color
        print(self.colorLower)
        print(self.colorUpper)

    def getpos(self, event: int, x: int, y: int, flags: Any, param: Any):
        """
        获取鼠标点击到的位置的坐标值
        Args:
            event:
            x:
            y:
            flags:
            param:

        Returns:

        """
        if event == cv2.EVENT_LBUTTONDOWN:  # 定义一个鼠标左键按下去的事件
            print(self.HSV[y, x])
            if self.HSV[y, x][0] < self.Hmin:
                self.Hmin = self.HSV[y, x][0]
            if self.HSV[y, x][0] > self.Hmax:
                self.Hmax = self.HSV[y, x][0]
            if self.HSV[y, x][1] < self.Smin:
                self.Smin = self.HSV[y, x][1]
            if self.HSV[y, x][1] > self.Smax:
                self.Smax = self.HSV[y, x][1]
            if self.HSV[y, x][2] < self.Vmin:
                self.Vmin = self.HSV[y, x][2]
            if self.HSV[y, x][2] > self.Vmax:
                self.Vmax = self.HSV[y, x][2]

    def setcolorvalue(self, color, color_list_low: list, color_list_up: list):
        """
        设置颜色阈值
        Args:
            color:
            color_list_low:
            color_list_up:

        Returns:

        """
        self.color_list_lower = color_list_low
        self.color_list_upper = color_list_up
        self.colorLower = np.array(self.color_list_lower)  # 这是红色的数值
        self.colorUpper = np.array(self.color_list_upper)
        self.color = color
        # print('设置阈值成功，当前阈值为：', self.color_list_lower, self.color_list_upper)

    def color_detect(self):
        """
        颜色检测
        Returns:

        """
        frame = np.copy(self.img)
        self.data = 'none'
        self.frame = frame
        self.img_new = frame
        # 转到HSV空间
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow('hsv',hsv)
        # cv2.waitKey(40)
        # 根据阈值构建掩膜
        mask = cv2.inRange(hsv, self.colorLower, self.colorUpper)
        #         cv2.imshow('mask_original', mask)
        #         cv2.waitKey(40)
        # 腐蚀操作
        mask = cv2.erode(mask, None, iterations=2)
        # 膨胀操作，其实先腐蚀再膨胀的效果是开运算，去除噪点
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        # 初始化识别物体圆形轮廓质心
        center = None
        # 如果存在轮廓
        if len(cnts) > 0:
            # 找到面积最大的轮廓
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)  # 最大面积区域的外接矩形   x,y是左上角的坐标，w,h是矩形的宽和高
            # print('x,y,w,h',x,y,w,h)
            if w > 60 and h > 60:  # 宽和高大于一定数值的才要。
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 255), 2)
                # print('{0}',format(x))
                if x < 0 or y < 0:
                    # self.img_new=frame_bgr
                    self.img_new = frame
                else:
                    self.img_new = frame[y + 5:y + h - 5, x + 5:x + w - 5]

                self.img_new = cv2.resize(self.img_new, dsize=(640, 480))  # 这一行是放大图像变回 640✖480
                cv2.imshow('result', self.img_new)
                cv2.waitKey(3)

            # 确定面积最大的轮廓的外接圆
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            self.x = x
            self.y = y
            self.radius = radius
            # 计算轮廓的矩
            M = cv2.moments(c)
            # 计算质心
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # 只有当半径大于100mm时，才执行画图
            if radius > 5:
                # img_circle=cv2.circle(self.frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                # cv2.circle(self.frame, center, 5, (0, 0, 255), -1)
                # 把质心添加到pts中，并且是添加到列表左侧
                self.pts.appendleft(center)
                # cv2.imshow('color1', self.frame)
                # cv2.waitKey(1)
                self.color_data = self.color
                X = x + w / 2
                Y = y + h / 2  # 中心点坐标
                self.distance = ((X - 250) ** 2 + (Y - 250) ** 2) ** 0.5  # 离中心点的距离
                self.midle = [X, Y]  # 检测后返回红点的中心值
        else:  # 如果图像中没有检测到识别物体，则清空pts，图像上不显示轨迹。
            self.pts.clear()
            # cv2.imshow('color2', self.frame)
            # cv2.waitKey(1)
            self.color_data = 'other_color'
            self.midle = [0, 0]  # 中心点坐标
            self.distance = 0  # 离中心点的距离

    """
    以下是物体形状检测与识别
    """

    def color_mask_init(self, mask_color: str):
        """
        物体形状检测与识别
        Args:
            mask_color:

        Returns:

        """
        if mask_color == 'red':
            self.mask_color_list_lower = [131, 56, 73]  # 这是红色的数值
            self.mask_color_list_upper = [180, 255, 255]
        elif mask_color == 'green':
            self.mask_color_list_lower = [18, 45, 97]
            self.mask_color_list_upper = [77, 255, 255]
        elif mask_color == 'yellow':
            self.mask_color_list_lower = [26, 43, 46]
            self.mask_color_list_upper = [34, 255, 255]
        elif mask_color == 'blue':
            self.mask_color_list_lower = [92, 56, 139]
            self.mask_color_list_upper = [102, 255, 255]
        elif mask_color == 'orange':
            self.mask_color_list_lower = [11, 43, 46]
            self.mask_color_list_upper = [25, 255, 255]
        elif mask_color == 'black':
            self.mask_color_list_lower = [0, 0, 0]
            self.mask_color_list_upper = [180, 255, 46]
        elif mask_color == 'white':
            self.mask_color_list_lower = [0, 0, 221]
            self.mask_color_list_upper = [180, 30, 255]
        elif mask_color == 'gray':
            self.mask_color_list_lower = [0, 0, 46]
            self.mask_color_list_upper = [180, 43, 220]
        elif mask_color == 'purple':
            self.mask_color_list_lower = [125, 43, 46]
            self.mask_color_list_upper = [155, 255, 255]
        elif mask_color == 'qing':
            self.mask_color_list_lower = [78, 43, 46]
            self.mask_color_list_upper = [99, 255, 255]
        self.mask_color_lower = np.array(self.mask_color_list_lower)
        self.mask_color_upper = np.array(self.mask_color_list_upper)

    def color_mask(self):
        """
        颜色掩膜
        Returns:

        """
        frame = np.copy(self.img)
        # 转到HSV空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow('hsv',hsv)
        # cv2.waitKey(40)
        # 根据阈值构建掩膜
        mask = cv2.inRange(hsv, self.mask_color_lower, self.mask_color_upper)
        #         cv2.imshow('mask_original', mask)
        #         cv2.waitKey(40)
        # 腐蚀操作
        mask = cv2.erode(mask, None, iterations=2)
        # 膨胀操作，其实先腐蚀再膨胀的效果是开运算，去除噪点
        self.mask_img = cv2.dilate(mask, None, iterations=2)
        self.mask_img = cv2.resize(self.mask_img, dsize=(640, 480))  # 这一行是放大图像变回 640✖480
        cv2.imshow('mask', self.mask_img)
        cv2.waitKey(3)

    def circle_detect(self, img: np.ndarray):
        """
        形状检测
        Args:
            img:

        Returns:

        """
        gaussian = cv2.GaussianBlur(img, (3, 3), 0)
        circles1 = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 110, param1=200, param2=35, minRadius=0,
                                    maxRadius=0)
        if circles1 is not None:
            self.shape_type = 'circle'
            circles = circles1[0, :, :]
            circles = np.uint16(np.around(circles))
            # print(circles)
            # print("圆心：", circles[0][0], circles[0][1])
            # print("半径：", circles[0][2])
            self.circle_center = [int(circles[0][0]), int(circles[0][1])]
            self.radium = int(circles[0][2])
            cv2.circle(self.img, self.circle_center, self.radium, (48, 48, 255), 2)
        else:
            self.shape_type = 'None'
            self.circle_center = [0, 0]
            self.radium = 0

    def polygon_detect(self):
        """
        多边形检测
        Returns:

        """
        contours, hierarchy = cv2.findContours(self.mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            # print('c', c)
            x0, y0, w0, h0 = cv2.boundingRect(c)  # 最大面积区域的外接矩形   x,y是左上角的坐标，w,h是矩形的宽和高
            if w0 > 40 and h0 > 40:  # 宽和高大于一定数值的才要。
                # 轮廓逼近
                epsilon = 0.0115 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                # 分析几何形状
                self.polygon_corners = len(approx)
                if self.draw_or_not:
                    x1, y1, w1, h1 = cv2.boundingRect(approx)
                    self.circle_center = [int((2 * x1 + w1) / 2), int((2 * y1 + h1) / 2)]
                    self.radium = int(math.sqrt((w1 / 2) ** 2 + (h1 / 2) ** 2) - 15)
                    cv2.circle(self.img, self.circle_center, self.radium, (48, 48, 255), 2)
                # 求解中心位置
                mm = cv2.moments(c)
                self.cx = int(mm['m10'] / (mm['m00'] + 1))
                self.cy = int(mm['m01'] / (mm['m00'] + 1))
                # 计算面积与周长
                self.length = cv2.arcLength(c, True)
                self.area = cv2.contourArea(c)
            else:
                self.radium = 0
                self.circle_center = [0, 0]

    def shape_detect(self, shape: str):
        """
        形状检测
        Args:
            shape:

        Returns:

        """
        self.color_mask()
        if shape == 'circle':
            mask = self.mask_img
            # gaussian = cv2.GaussianBlur(mask, (3, 3), 0)
            edge = cv2.Canny(mask, 30, 100)
            self.circle_detect(edge)
        if shape in ['triangle', 'rectangle', 'polygon']:
            self.polygon_detect()
            if self.polygon_corners == 3 and shape == 'triangle':
                self.shape_type = "triangle"
                self.draw_or_not = 1
            elif 4 <= self.polygon_corners <= 8 and shape == 'rectangle':
                self.shape_type = "rectangle"
                self.draw_or_not = 1
            elif self.polygon_corners >= 9 and shape == 'polygon':
                self.shape_type = "polygon"
                self.draw_or_not = 1
            else:
                self.shape_type = 'None'
                self.draw_or_not = 0
        self.shape_position()

    def shape_analysis(self):
        """
        得到识别到的形状
        Returns:

        """
        self.polygon_detect()
        if self.polygon_corners == 3:
            self.shape_type = "triangle"
        elif 4 <= self.polygon_corners <= 9:
            self.shape_type = "rectangle"
        elif self.polygon_corners >= 10:
            self.shape_type = "polygon"
        else:
            self.shape_type = 'None'

    def shape_position(self):
        """
        返回识别到的形状的坐标
        Returns:

        """
        center = self.circle_center
        b, a, channels = self.img.shape
        if 0 < center[0] <= a / 3:
            self.shape_direction_p = 'left'
        elif a / 3 < center[0] < 2 * a / 3:
            self.shape_direction_p = 'middle'
        elif 2 * a / 3 < center[0] < a:
            self.shape_direction_p = 'right'
        if 0 < center[1] <= b / 3:
            self.shape_direction_s = 'top'
        elif b / 3 < center[1] < 2 * b / 3:
            self.shape_direction_s = 'middle'
        elif 2 * b / 3 < center[1] < b:
            self.shape_direction_s = 'right'

    """
    以下为颜色聚类,涉及到sklearn的聚类模型
    """

    def color_cluster_init(self):
        self.color_cluster_data = []
        self.is_break = False
        self.is_recording = False
        cv2.namedWindow('img')
        cv2.setMouseCallback("img", self.mouse)
        shutil.rmtree(color_cluster_path)
        os.mkdir(color_cluster_path)

    def color_cluster_get(self):
        while True:
            self.get_img()
            copy_img = self.img.copy()
            cv2.rectangle(self.img, (490, 5), (630, 50), (0, 255, 0), thickness=-1)
            cv2.rectangle(self.img, (490, 430), (630, 475), (177, 177, 140), thickness=-1)
            draw_dotted_rect(self.img, (192, 144), (448, 336), (255, 255, 0), 3)
            num = len(self.color_cluster_data)
            self.img = cv2AddChineseText(self.img, '请将物品覆盖整个方框', (10, 10), textSize=25)
            self.img = cv2AddChineseText(self.img, '已采集图片:{}'.format(num), (300, 10), textColor=(0, 0, 255),
                                         textSize=25)
            self.img = cv2AddChineseText(self.img, '完成', (530, 440), textColor=(255, 0, 0), textSize=25)
            if self.is_recording:
                self.img = cv2AddChineseText(self.img, '正在采集中……', (10, 50), textSize=25)
                self.img = cv2AddChineseText(self.img, '停止采集图片', (502, 16), textColor=(0, 0, 0), textSize=20)
                self.color_cluster_data.append(copy_img[144:336, 192:448])
                # cv2.imwrite(os.path.join(color_cluster_path ,str(num + 1) ,'.jpg'), copy_img[144:336, 192:448])
                cv2.waitKey(20)
            else:
                self.img = cv2AddChineseText(self.img, '开始采集数据', (502, 16), textColor=(0, 0, 0), textSize=20)
            if self.is_break:
                np.save(os.path.join(color_cluster_path, 'color_cluster.npy'), self.color_cluster_data)
                print("数据采集成功！")
                break
            cv2.imshow('img', self.img)
            cv2.waitKey(1)

    def color_cluster_train(self, cluster_num, name):
        from sklearn.model_selection import train_test_split
        from sklearn.cluster import KMeans
        import joblib
        input_data = []
        if not os.path.exists(os.path.join(color_cluster_path, 'color_cluster.npy')):
            raise FileNotFoundError('你还没有收集足够的数据！')
        data = np.load(os.path.join(color_cluster_path, 'color_cluster.npy'))
        for i in data:
            B = i[:, :, 0].flatten()
            G = i[:, :, 1].flatten()
            R = i[:, :, 2].flatten()
            input_data.append(np.array([B.mean(), G.mean(), R.mean()]))
        input_data = np.array(input_data)
        print('开始训练……')
        k_means = KMeans(n_clusters=cluster_num, max_iter=300)
        k_means.fit(input_data)
        joblib.dump(k_means, os.path.join(model_path, name + '.model'))
        print('模型训练成功！')

    def color_cluster_predict(self, name):
        import joblib
        if not os.path.exists(os.path.join(model_path, name + '.model')):
            raise FileNotFoundError('你还没有开始训练名为' + name + '的模型！')
        k_means = joblib.load(os.path.join(model_path, name + '.model'))
        middle_part = self.img[144:336, 192:448]
        B = middle_part[:, :, 0].flatten()
        G = middle_part[:, :, 1].flatten()
        R = middle_part[:, :, 2].flatten()
        input_data = np.array([B.mean(), G.mean(), R.mean()]).reshape(1, 3)
        self.color_cluster_result = k_means.predict(input_data)[0]

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if 630 > x > 490 and 5 < y < 50:
                self.is_recording = not self.is_recording
            if 630 > x > 490 and 430 < y < 475:
                self.is_break = True

    """
    以下为mediapipe的人体姿态交互功能的函数，为保证最大自由度，函数可返回读取的人体关键点数值，其中有些部分的函数的返回值用户可以自行选择。
    """

    def finger_init(self):
        """
        初始化手指检测器
        Returns:

        """
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def finger_detect(self):
        """
        开始手指检测
        Returns:

        """
        img_new = self.img.copy()
        h, w, _ = img_new.shape
        imgRGB = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for id, handlms in enumerate(results.multi_hand_landmarks):
                self.fingertip["big_finger" + (str(id) if id != 0 else '')] = (
                    int(handlms.landmark[4].x * w), int(handlms.landmark[4].y * h))
                self.fingertip["fore_finger" + (str(id) if id != 0 else '')] = (
                    int(handlms.landmark[8].x * w), int(handlms.landmark[8].y * h))
                self.fingertip["middle_finger" + (str(id) if id != 0 else '')] = (
                    int(handlms.landmark[12].x * w), int(handlms.landmark[12].y * h))
                self.fingertip["ring_finger" + (str(id) if id != 0 else '')] = (
                    int(handlms.landmark[16].x * w), int(handlms.landmark[16].y * h))
                self.fingertip["little_finger" + (str(id) if id != 0 else '')] = (
                    int(handlms.landmark[20].x * w), int(handlms.landmark[20].y * h))
            self.mpDraw.draw_landmarks(img_new, handlms, self.mpHands.HAND_CONNECTIONS)
        else:
            self.fingertip = {}
        img_new = cv2.resize(img_new, dsize=(640, 480))  # 这一行是放大图像变回 640✖480
        cv2.imshow('finger_detect', img_new)

    """
    手指检测
    """

    def finger_distance(self, tip1: str = 'big_finger', tip2: str = 'fore_finger'):  # 返回用户选定的两个手骨关键点的距离
        """
        返回用户选定的两个手骨关键点的距离
        Args:
            tip1:
            tip2:

        Returns:

        """
        if not bool(self.fingertip):
            return -1
        if tip2 == tip1:
            return 0
        else:
            return math.pow(math.pow(self.fingertip[tip1][0] - self.fingertip[tip2][0], 2) + math.pow(
                self.fingertip[tip1][1] - self.fingertip[tip2][1], 2), 0.5)

    """
    身体部位检测
    """

    def body_init(self):
        """
        身体部位检测初始化
        Returns:

        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

    def body_detect(self):
        """
        开始身体部位检测
        Returns:

        """
        img_new = self.img.copy()
        h, w, _ = img_new.shape
        img_new = self.img.copy()
        imgRGB = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        if results.pose_landmarks:
            poselms = results.pose_landmarks.landmark
            for tur in (
                    ('left_wrist', 15), ('right_wrist', 16), ('left_elbow', 13), ('right_elbow', 14),
                    ('left_ankle', 27),
                    ('right_ankle', 28), ('left_shoulder', 11), ('right_shoulder', 12)):
                self.body_menu[tur[0]] = (int(poselms[tur[1]].x * w), int(poselms[tur[1]].y * h))
            self.mpDraw.draw_landmarks(img_new, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        img_new = cv2.resize(img_new, dsize=(640, 480))  # 这一行是放大图像变回 640✖480
        cv2.imshow('body_detect', img_new)

    def wrist_mark(self, wrist: str = 'left_wrist', mark: str = 'x'):
        """
        返回左手或右手的检测参数
        Args:
            wrist:
            mark:

        Returns:

        """
        if not bool(self.body_menu):
            return -1
        return self.body_menu[wrist][0 if mark == 'x' else 1]

    def wrist_distance(self, wrist1: str = 'left_wrist', wrist2: str = 'right_wrist'):
        """
        返回选定的两个关节之间的距离，没有识别到就返回-1
        Args:
            wrist1:
            wrist2:

        Returns:

        """
        if not bool(self.body_menu):
            return -1
        if wrist1 == wrist2:
            return 0
        return math.pow(math.pow(self.body_menu[wrist1][0] - self.body_menu[wrist2][0], 2) + math.pow(
            self.body_menu[wrist1][1] - self.body_menu[wrist2][1], 2), 0.5)

    """
    背景切换
    """

    def backCroundChange_init(self):
        """
        初始化背景切换器
        Returns:

        """
        self.segmentor = SelfiSegmentation()

    def backGroundChange(self, backGroundImageSrc: str = "1.png"):
        """
        切换背景
        Args:
            backGroundImageSrc:

        Returns:

        """
        backGroundImageSrc = (picture_path if os.path.isabs(backGroundImageSrc) == False else '') + backGroundImageSrc
        img_new = self.img.copy()
        h, w, _ = img_new.shape
        backGroundImage = cv2.resize(cv2.imread(backGroundImageSrc), (w, h), cv2.INTER_AREA)

        img_new = self.segmentor.removeBG(img_new, backGroundImage, threshold=0.5)
        img_new = cv2.resize(img_new, dsize=(640, 480))  # 这一行是放大图像变回 640✖480
        cv2.imshow('backGroundChange', img_new)

    def faceMeshDetect_init(self):
        """
        初始化人脸特征点检测器
        Returns:

        """
        self.mpFaceMesh = mp.solutions.face_mesh
        self.mpDraw = mp.solutions.drawing_utils
        self.FaceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=2)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def faceMesh(self):
        """
        开始检测人脸特征点
        Returns:

        """
        img_new = self.img.copy()
        imgRGB = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)
        results = self.FaceMesh.process(imgRGB)
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img_new, faceLms, self.mpFaceMesh.FACEMESH_FACE_OVAL,
                                           landmark_drawing_spec=self.drawSpec)
        img_new = cv2.resize(img_new, dsize=(640, 480))  # 这一行是放大图像变回 640✖480
        cv2.imshow('faceMesh', img_new)

    def get_shape(self, parameter: str = 'width'):
        """
        用于获取图像的尺寸，并根据用户的选择来返回对应的或长或高或是通道数
        Args:
            parameter:

        Returns:

        """

        try:
            h, w, c = self.img.shape
        except(ValueError):
            h, w = self.img.shape
            c = 1
        self.parameter = str(parameter)
        shape = {'height': h, 'width': w, 'channel': c}
        return shape[self.parameter]

    """
    模版匹配
    """

    def match_template_init(self):
        """
        初始化模板匹配器
        Returns:

        """
        self.template = None
        self.min_val, self.max_val, self.min_loc, self.max_loc = None, None, None, None
        self.isshow = False
        templist = []
        for i in range(0, 9):
            im3 = cv2.imread(picture_path + 'blue' + str(i) + '.jpg')
            im3 = cv2.resize(im3, dsize=(99, 116))
            templist.append(im3)

    def match_template(self, template_name: str):
        """
        开始模板匹配
        Args:
            template_name:

        Returns:

        """
        self.isshow = False
        self.template = cv2.imread((picture_path if os.path.isabs(template_name) == False else '') + template_name)
        # print(self.template.shape)
        self.template = cv2.resize(self.template, (99, 116))
        result = cv2.matchTemplate(self.img, self.template, cv2.TM_CCOEFF_NORMED)
        self.min_val, self.max_val, self.min_loc, self.max_loc = cv2.minMaxLoc(result)

    def match_result(self):
        """
        返回模板匹配的结果
        Returns:

        """
        self.isshow = True
        w, h = self.template.shape[1], self.template.shape[0]
        top_left = self.max_loc
        self.max_loc_topleft = list(self.max_loc)
        self.max_loc_bottomright = [top_left[0] + w, top_left[1] + h]
        bottom_right = [top_left[0] + w, top_left[1] + h]
        self.cut_img = self.img[top_left[1]:top_left[1] + h, top_left[0]: top_left[0] + w]
        self.cut_img = cv2.resize(self.img, dsize=(640, 480))  # 这一行是放大图像变回 640✖480
        if self.isshow:
            cv2.imshow('cut', self.cut_img)
            cv2.rectangle(self.img, top_left, bottom_right, 255, 2)
        """
    def predict_number_match(self, input):
        for t in templist:
            res = cv2.matchTemplate(input, t, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            score.append(max_val)
        self.predict = np.argmax(score)
        self.predict = np.argmax(score)
        """
