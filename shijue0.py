import os
import cv2
import imutils
import numpy as np
import pyzbar.pyzbar as pyzbar
from typing import Any
from .util.all_path import picture_path, system_platform


def new_file(path: str):
    """
    新建文件夹
    Args:
        path:

    Returns:

    """
    if os.path.isdir(path):
        pass
    else:
        print('未找到该文件夹。开始创建文件夹')
        os.makedirs(path)


def pic_read(filename: str, mode: int):
    """
        读取图片。
    :param filename:  图片的路径包括名称
    :param mode:  读取的模式，彩色还是灰度
    :return: 返回读取的图片
    其实严格来说，不是imread不支持中文路径，而是不支持non-ascii。
    所以不论路径如何转换编码格式，应该都不能解决问题。
    解决的思路就是先用其他支持中文的API，把图片数据导入到内存中，
    然后通过opencv从内存读入图片的方法，读入图片
    """
    raw_data = np.fromfile(filename, dtype=np.uint8)
    img = cv2.imdecode(raw_data, mode)
    return img


class basicImg:
    """
    基础图像操作类
    """

    def __init__(self):
        """
        默认属性都放在此处
        """
        self.img = None
        self.cam = None
        """
        二维码属性放在此处
        """
        self.er_data = 'none'
        self.QR_code_data = None
        """
        这里的图像设置了是使用500*500像素，可以自己改，或者保留原来的树莓派版本的。
        """
        self.midle = [0, 0]  # 颜色检测框出的中心点
        self.distance = 0  # 与中心点的距离
        self.picture = 0  # 用来判断是否是图像，用于后面显示图像自动加入waitKey并且让图像不放大两倍
        # self.picture_img = 0  #用来防止同时使用摄像头读取图像与从路径读取图片，如果等于了2就会print无法使用。考虑到运行速度问题，取消这个参数，不要过多的if判断了

    """
        这是视觉专用的库
        shijue0为基本操作
        shijue1为高级操作
    """

    # 获取摄像头
    def camera(self, num: int = 0):
        """
        获取摄像头
        Args:
            num:

        Returns:

        """
        if 'win' in system_platform:
            self.cam = cv2.VideoCapture(num, cv2.CAP_DSHOW)
        else:
            self.cam = cv2.VideoCapture(num)
            # 下面两行设置了摄像头分辨率为320✖240，这样处理不会那么卡
            self.cam.set(3, 320)
            self.cam.set(4, 240)
        # 如果是在Windows就不改变摄像头分辨率（PC算力足够）

    def close_camera(self):
        """
        关闭摄像头
        Returns:

        """
        self.cam.release()

    # get_img 是用来获取单张图片的
    def get_img(self):
        """
        获取图片
        Returns:

        """
        self.ret, img = self.cam.read()
        if self.ret:
            self.img = img
        else:
            print("未检测到摄像头，请注意摄像头是否接触不良或者未设置允许摄像头")

    # img_flip 是用来翻转镜像图片的
    def img_flip(self, flip_by: str):
        """
        图像翻转
        Args:
            flip_by:

        Returns:

        """
        if flip_by == 'y':
            self.img = cv2.flip(self.img, 1)
        elif flip_by == 'x':
            self.img = cv2.flip(self.img, 0)
        elif flip_by == 'xy':
            self.img = cv2.flip(self.img, -1)

    # get_frame 是从某个路径中获取图片
    def get_frame(self, path: str, file_dir: str):
        """
        从某个路径中获取图片
        :param path:  用户传进来的文件名，可能是相对路径或绝对路径
        :param file_dir:  如果是相对路径，就需要用到这个文件路径
        :return:
        """
        file_dir = file_dir[:file_dir.rfind("\\")]

        if os.path.isabs(path) == False:
            if os.path.exists(os.path.join(picture_path, path)):
                path = os.path.join(picture_path, path)
            elif os.path.exists(os.path.join(file_dir, path)):
                path = os.path.join(file_dir, path)
        self.img = pic_read(path, cv2.IMREAD_COLOR)
        self.picture = 1  # 如果有运行从路径读取图片，赋值这个参数为 1

    # name_windows 是用来命名图片展示窗口的
    def name_windows(self, name: str):
        """
        命名图片展示窗口的
        Args:
            name:

        Returns:

        """
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)  # cv2.WINDOW_AUTOSIZE 窗口不可拉伸    cv2.WINDOW_NORMAL 窗口可以随意拉伸

    # close_windows 是用来关闭所有窗口的
    def close_windows(self):
        """
        关闭所有窗口
        Returns:

        """
        cv2.destroyAllWindows()

    # show_image是用来将图片展示在定义的某个窗口中的
    def show_image(self, windows_name: str):
        """
        显示图像
        Args:
            windows_name:

        Returns:

        """
        # img = cv2.resize(self.img, dsize=(640, 480))  # 这一行是放大图像变回 640✖480
        cv2.imshow(windows_name, self.img)

    # write_image 是用来保存图片的函数，注意：pic_name中不能存在中文，包括路径和文件命名
    def write_image(self, pic_name: str, jpg_png: np.ndarray = False):
        """
        保存图像
        Args:
            pic_name:
            jpg_png:

        Returns:

        """
        if pic_name.split(".")[-1] not in ["jpg", "png"]:
            raise NameError("图片名字缺少后缀jpg或者png或者后缀不对,请使用xx.jpg或xx.png这种名字来保存")

        if os.path.isabs(pic_name):  # 如果是绝对路径，那就不修改
            path = pic_name
        else:  # 如果是相对路径，那就补成绝对路径
            path = os.path.join(picture_path, pic_name)

        # path = picture_path + pic_name + mode
        result = cv2.imwrite(path, self.img)
        if result == False:
            raise NameError("保存失败，文件名不能含有中文")
        print('图片已保存到：', path)

    # resize 是用来改变图像的大小的
    def resize(self, newsize: tuple = (1, 1)):
        """
        改变图像大小
        Args:
            newsize:

        Returns:

        """
        self.img = cv2.resize(self.img, newsize)

    # delay 是用来进行展示延时的，有一个或者多个窗口进行展示时，此函数必须用上
    def delay(self, time: int = 1):
        """
        延时，为了展示图像，这个函数是必须的
        Args:
            time:

        Returns:

        """
        try:
            cv2.waitKey(time)
        except KeyboardInterrupt:
            pass

    # erosion 是用给图片进行腐蚀操作的
    def erosion(self):
        """
        图像腐蚀
        Returns:

        """
        self.img = cv2.erode(self.img, None, iterations=2)

    # dilation 是用来给图片进行膨胀操作的
    def dilation(self):
        """
        图像膨胀
        Returns:

        """
        self.img = cv2.dilate(self.img, None, iterations=2)

    # BGR2GRAY是用来将彩色图转成灰度图的
    def BGR2GRAY(self):
        """
        彩色图转成灰度图
        Returns:

        """
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    # GRAY2BIN 是用来将灰度图转成二值化图的
    def GRAY2BIN(self):
        """
        是用来将灰度图转成二值化图的
        Returns:

        """
        _, self.img = cv2.threshold(self.img, 0, 255, cv2.THRESH_OTSU)

    def canny(self):
        """
        边缘检测
        Returns:

        """
        canny_img = cv2.Canny(self.img, 30, 100)
        cv2.imshow('canny', canny_img)

    def find_Contour(self):
        """
        查找轮廓
        Returns:

        """
        if self.img.ndim == 2:
            raise ValueError('imgTypeError: 输入的图片必须是彩色图像')
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(self.img, contours, -1, (255, 0, 0), 3)

    def img_type(self):
        """
        返回图像类型，彩色还是灰度
        Returns:

        """
        return 'RGB彩色图像' if self.img.ndim == 3 else '灰度图'

    """
        以下 decodeDispaly、erweima_detect函数是用来实现扫描二维码功能的
        __decodeDisplay: 解码
        erweima_detect:进行二维码检测
        示例：
        I = Img()
        I.camera(0)
        I.name_windows('img')
        while True:
            I.get_img()
            I.erweima_detect()
            I.show_image('img')
            print(I.QR_code_data)
            I.delay(1)
    """

    def __decodeDisplay(self, image: np.ndarray):  # 解码部分
        """
        解码
        Args:
            image:

        Returns:

        """
        barcodes = pyzbar.decode(image)
        img = self.img
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)
            cv2.circle(img, (int(x + w / 2), int(y + h / 2)), int(h / 2), (255, 0, 0), 5)
            barcodeData = barcode.data.decode("utf-8")
            self.er_data = barcodeData
            self.QR_code_data = self.er_data
        # self.name_windows('Result of QRcode')
        # self.show_image('Result of QRcode', img)
        img = cv2.resize(self.img, dsize=(640, 480))
        cv2.imshow('Result of QRcode', img)

    def erweima_detect(self):
        """
        扫描二维码
        Returns:

        """
        img = self.img
        self.er_data = 'none'
        self.QR_code_data = self.er_data
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.__decodeDisplay(gray)

    """
        beauty_face 函数是用来磨皮的
    """

    def beauty_face(self):
        """
        美颜磨皮
        Returns:

        """
        v1 = 3  # 磨皮程度
        v2 = 2  # 细节程度
        dx = v1 * 5  # 双边滤波参数之一
        fc = v1 * 12.5  # 双边滤波参数之一
        p = 0.1
        # 双边滤波
        copy = self.img
        temp1 = cv2.bilateralFilter(copy, dx, fc, fc)
        temp2 = cv2.subtract(temp1, copy)
        temp2 = cv2.add(temp2, (10, 10, 10, 128))
        # 高斯模糊
        temp3 = cv2.GaussianBlur(temp2, (2 * v2 - 1, 2 * v2 - 1), 0)
        temp4 = cv2.add(copy, temp3)
        dst = cv2.addWeighted(copy, p, temp4, 1 - p, 0.0)
        img = cv2.add(dst, (10, 10, 10, 255))
        self.name_windows('after_beauty')
        # self.show_image('after_beauty', img)
        img = cv2.resize(self.img, dsize=(640, 480))
        cv2.imshow('after_beauty', img)

    """
        以下是彬锋师兄于2021年初写的视觉函数，用于重庆杯的比赛内容
    """

    # 二值化图寻迹
    def offset_calculate1(self, y: int = -1, img: np.ndarray = []):
        """
        二值化图寻迹
        Args:
            y:
            img:

        Returns:

        """
        if len(img) == 0:
            img = self.img
        if -1 == y:
            y = img.shape[0]
            y = y // 2
        line = img[y]
        white_count = np.sum(line == 0)
        white_index = np.where(line == 0)
        if white_count == 0:
            return 0
        center = (white_index[0][white_count - 1] + white_index[0][0]) / 2
        # 求图像中心
        img_width = self.img.shape[1]
        img_center = img_width // 2
        direction = center - img_center
        direction = int(direction)
        return direction

    def line_angle1(self, img: np.ndarray = []):
        """
        获取图像的角度
        Args:
            img:

        Returns:

        """
        if len(img) == 0:
            img = self.img
        h = img.shape[0]
        up = []
        down = []
        for i in range(0, h // 2, 10):
            up.append(self.offset_calculate1(i))
        for i in range(h // 2, h, 10):
            down.append(self.offset_calculate1(i))
        a = sum(up) // len(up)
        b = sum(down) // len(down)
        angle = (a - b) // 180
        return angle

    def offset1(self, img: np.ndarray = []):
        """
        判断图像黑线是否在中心
        Args:
            img:

        Returns:

        """
        if len(img) == 0:
            img = self.img
        y = img.shape[0]
        line = img[y // 2]
        white_count = np.sum(line == 0)
        white_index = np.where(line == 0)
        if white_count == 0:
            return False
        else:
            return True

    def dotted_line1(self, img: np.ndarray = []):
        """
        画线
        Args:
            img:

        Returns:

        """
        if len(img) == 0:
            img = self.img
        cnts = self.bin_detect(img)
        x = []
        for c in cnts:
            area = self.cnt_area(c)
            M = cv2.moments(c)
            if area > 3000:
                cx = int(M["m10"] / M["m00"])
                x.append(cx)
        if len(x) < 2:
            return False
        a = np.std(np.array(x))
        if a > 150:
            return False
        else:
            return True

    # 彩色图寻迹
    def offset_calculate2(self):
        """
        彩色图寻迹
        Returns:

        """
        gray = cv2.cvtColor(self.img.copy(), cv2.COLOR_BGR2GRAY)
        retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        img = cv2.dilate(dst, None, iterations=2)
        h = img.shape[0]
        direction = self.offset_calculate1(h // 2, img)
        return direction

    def line_angle2(self):
        """
        未知
        Returns:

        """
        gray = cv2.cvtColor(self.img.copy(), cv2.COLOR_BGR2GRAY)
        retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        img = cv2.dilate(dst, None, iterations=2)
        angele = self.line_angle1(img)
        return angele

    def offset2(self):
        """
        膨胀之后再判断中心黑线
        Returns:

        """
        gray = cv2.cvtColor(self.img.copy(), cv2.COLOR_BGR2GRAY)
        retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        img = cv2.dilate(dst, None, iterations=2)
        result = self.offset1(img)
        return result

    def dotted_line2(self):
        """
        画线2
        Returns:

        """
        gray = cv2.cvtColor(self.img.copy(), cv2.COLOR_BGR2GRAY)
        retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        img = cv2.dilate(dst, None, iterations=2)
        result = self.dotted_line1(img)
        return result

    def cnt_area(self, cnt: Any):
        """
        返回轮廓面积
        Args:
            cnt:

        Returns:

        """
        area = cv2.contourArea(cnt)
        return area

    def detect(self, c: Any, Shape: str):
        """
        定义形状名称和判断近似形状
        Args:
            c:
            Shape:

        Returns:

        """
        shape = "未知形状"
        peri = cv2.arcLength(c, True)  # 周长
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        r1 = peri / 6.2  # 半径
        area = cv2.contourArea(c)  # 面积
        r2 = (area / 3.14) ** 0.5
        if abs(r1 - r2) < 0.22 * r1 and len(approx) > 4:
            shape = "圆形"
        elif len(approx) == 3:
            shape = "三角形"
        # 判断四边形是正方形还是长方形
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "正方形" if ar >= 0.95 and ar <= 1.05 else "长方形"
        else:
            pass
        return (shape == Shape)

    def bin_detect(self, img: np.ndarray = []):
        """
        在阈值图像中查找轮廓并初始化形状检测器
        Args:
            img:

        Returns:

        """
        if len(img) == 0:
            img = self.img
        # 在阈值图像中查找轮廓并初始化形状检测器
        cnts = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        max_cnts = []
        for c in cnts:
            area = self.cnt_area(c)
            if area > 2000:
                max_cnts.append(c)
        return max_cnts

    def cnt_draw(self, c: Any, shape: str):
        """
        标注识别到的形状
        Args:
            c:
            shape:

        Returns:

        """
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.drawContours(self.img, [c], -1, (0, 255, 0), 2)
        cv2.putText(self.img, shape, (cx, cy), 0, 2, (0, 255, 0), 3)

    def cnt_center(self, c: Any):
        """
        返回轮廓中心
        Args:
            c:

        Returns:

        """
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return [cx, cy]
