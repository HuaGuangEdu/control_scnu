#!/usr/local/bin/python3

'''
这是用来控制硬件的源码程序
'''

from __future__ import division
import time
import sys
import cv2

'''
这下面是屏幕需要的新库
'''
import os
import logging  #加载字体
from PIL import Image, ImageDraw, ImageFont, ImageFilter  #放图片的包

system_platform = sys.platform
if 'win' in system_platform:
    pass
else:
    try:
        from control.lcd import LCD_2inch4  # 屏幕的包
    except:
        raise '树莓派没有打开SPI接口！'
    import serial
    import Adafruit_DHT
    import RPi.GPIO as GPIO
    import Adafruit_PCA9685
    # 导入操控PCA9685芯片的库
    try:
        from adafruit_servokit import ServoKit
    except ImportError:
        try:
            os.system('sudo pip3 install adafruit-circuitpython-servokit')
        except:
            raise '没有连接网络，无法安装相关库！'

    # 设置编码格式为BCM，关闭警告
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

system_platform = sys.platform
main_path = '/home/pi/class/'  # 读取和保存文件所用主文件夹
if 'win' in system_platform:
    file_path = os.getcwd()
    # 获取当前文件的位置
    main_path = file_path + '/resources/assets/class/'
font_path = main_path + 'fonts/'

# 全局变量，麦克纳姆轮的初始速度
old_fb = 0
old_lr = 0
old_tn = 0


# 7月份钣金小车版本对应扩展板的 io_BCM字典
IO2GPIO = {0: 16, 1: 12, 2: 25,
           3: 24, 4: 22, 5: 23,
           6: 27, 7: 17, 8: 4}

PWM2GPIO = {0: 0, 1: 1,
            2: 2, 3: 3}

UW_IO = {1: [5, 6],
         2: [13, 19]}


# 初始类 Io
class Io(object):
    '''
    Introduction:
        Io类是用来调用拓展板上面IO口进行输入输出读取等操作的

    Attributes:
        io_num:IO口的编号，0-8
    '''

    def __init__(self, io_num: int):
        '''
        初始化Io类
        :param io_num:扩展板上IO口的编号，整数
        '''
        self.gpio_num = IO2GPIO[io_num]
        self.ioin = 404
        GPIO.setmode(GPIO.BCM)

    def set_io_mode(self, gpio_mode: str):
        '''
        设置GPIO的模式,bcm或者board
        :param gpio_mode: 设置GPIO的模式，BCM或BOARD，字符
        :return: None
        '''
        if gpio_mode == 'BCM':
            GPIO.setmode(GPIO.BCM)
        elif gpio_mode == 'BOARD':
            GPIO.setmode(GPIO.BOARD)
    
    def set_in_out(self, in_or_out: str):
        '''
        设置GPIO是输入或者输出模式
        :param in_or_out: 设置GPIO的输入或者输出模式，IN输入，OUT输出，字符
        :return: None
        '''
        if in_or_out == 'IN':
            GPIO.setup(self.gpio_num, GPIO.IN)
        elif in_or_out == 'OUT':
            GPIO.setup(self.gpio_num, GPIO.OUT)
    
    def set_io_out(self, dian_ping: str):
        '''
        设置GPIO输出高or低电平
        :param dian_ping: GPIO输出高or低电平，HIGH高电平，LOW低电平，字符类型
        :return: None
        '''
        if dian_ping == 'HIGH':
            GPIO.output(self.gpio_num, GPIO.HIGH)
        elif dian_ping == 'LOW':
            GPIO.output(self.gpio_num, GPIO.LOW)
    
    def get_io_in(self):
        '''
        获取GPIO口的输入电平,通过self.ioin获取
        :return: None
        '''
        # 获取GPIO口的输入电平
        if GPIO.input(self.gpio_num) == 0:
            # 低电平,返回false
            time.sleep(0.01)
            if GPIO.input(self.gpio_num) == 0:
                # 防抖设计
                self.ioin = 0
                # 输入为低电平
        else:
            self.ioin = 1
            time.sleep(0.01)
    
    def clean_io(self):
        '''
        清理IO口
        :return: None
        '''
        GPIO.cleanup(self.gpio_num)


class IoToPwm(object):
    '''
    Introduction:
        这是通过IO去控制PWM的类

    Attributes:
        freq: 输出的PWM频率，默认为50Hz
        duty：输出的PWM占空比，默认为50%

    Example:
        # 通过IO口输出pwm波
        p = GPIO.PWM(channel, frequency)
        p.start(dc) # where dc is the duty cycle (0.0 <= dc <= 100)
        p.ChangeFrequency(freq)  # freq 是以Hz为单位的新频率
        p.ChangeDutyCycle(dc)  # where 0.0 <= dc <= 100.0
        p.stop()
    '''
    def __init__(self, io_num: int, freq: int = 50, duty: int = 50):
        '''
        初始化 Io2Pwm 类
        :param io_num: 控制的IO口，指扩展板上的IO口
        :param freq: 输出的PWM频率，默认为50Hz
        :param duty: 输出的PWM占空比，默认为50%
        '''
        self.__pwm_io = IO2GPIO[io_num]
        GPIO.setup(self.__pwm_io, GPIO.OUT)
        self.freq = freq
        self.duty = duty
    
    def start(self):
        '''
        开始产生pwm
        :return:None
        '''
        self.__io_to_pwm = GPIO.PWM(self.__pwm_io, self.freq)
        self.__io_to_pwm.start(self.duty)
    
    def set_freq(self, pwm_freq: int):
        '''
        设置输出pwm波的频率
        :param pwm_freq: 输出的频率，数字，如50
        :return: None
        '''
        self.freq = pwm_freq
    
    def set_duty(self, pwm_duty: int):
        '''
        设置或改变输出pwm波的占空比，占空比输入范围：0-100
        :param pwm_duty: 输出的占空比，数字，如50
        :return: None
        '''
        self.duty = pwm_duty
        self.__io_to_pwm.ChangeDutyCycle(self.duty)
    
    def end(self):
        '''
        关闭pwm波
        :return: None
        '''
        self.__io_to_pwm.stop()


class PWM(object):
    '''
    Introduction:
        树莓派中控制PWM波的类

    Attributes:
        duty:pwm波的占空比
        freq:PWM波的频率
    '''
    def __init__(self, pwm_io: int):
        '''
        初始化 PWM 类
        :param pwm_io: 输入的pwm的口，有0-3 共四个口，整数类型
        '''
        # 初始化,必须提供是哪个pwm口，即打开pwm功能,pwm_io>40
        self.__pw = Adafruit_PCA9685.PCA9685()
        self.__io_pwm = PWM2GPIO[pwm_io]
        self.duty = 50
        self.freq = 262
        # 1:262 2:294 3:330 4:349 5:392 6:440 7:494
    
    '''
    pwm = Adafruit_PCA9685()
    def set_servo_angle(channel, angle):
        date = 4096*((angle*11)+500)/20000
        pwm.set_pwm(channel, 0, date)

    pwm.set_pwm_freq(50)
    while True:
        channel = int(input())
        angle = int(input())
        set_servo_angle(channel, angle)

    '''
    '''
        duty : 占空比
        freq : 频率
        self.io_pwm: 通道
    '''
    
    def pwm_start(self):
        '''
        开始产生PWM波
        :return:None
        '''
        # pw产生的PWM波发生器
        self.__pw.set_pwm(self.__io_pwm, 0, int((100 - self.duty) * 40.95))
        # 对于低电平有效的RGB灯而言
    
    def change_duty(self, duty: int):
        '''
        改变pwm波输出的占空比
        :param duty: 改变的占空比，整数，int
        :return:None
        '''
        self.duty = duty
        self.__pw.set_pwm(self.__io_pwm, 0, int((100 - self.duty) * 40.95))
    
    def change_freq(self, freq: int):
        '''
        改变pwm波输出的频率
        :param freq: 改变的频率，整数，int
        :return: None
        '''
        self.freq = freq
        self.__pw.set_pwm_freq(self.freq)
    
    def pwm_stop(self):
        '''
        停止PWM波
        :return: None
        '''
        self.__pw.set_pwm(self.__io_pwm, 0, int((100 - 0) * 40.95))


class CSB(object):
    '''
    Introduction:
        这是用来控制超声波的类，超声波接口有1，2共两个接口

    Attributes:
        dis:通过超声波计算出的距离
    Examples:

    '''
    def __init__(self, uw_num: int):
        '''
        初始化CSB类
        :param uw_num: 超声波接口，整数类型，如1
        '''
        self.__trig_p = UW_IO[uw_num][0]
        self.__echo_p = UW_IO[uw_num][1]
        self.dis = 0
        GPIO.setup(self.__trig_p, GPIO.OUT)
        GPIO.setup(self.__echo_p, GPIO.IN)
    
    '''
        self.trig_p: trig 对应的引脚
        self.echo_p: echo 对应的引脚
        self.dis   : 返回的距离
    '''
    
    def __sent_t_pulse(self):
        '''
        开始发送超声波
        :return: None
        '''
        # 发送超声波
        GPIO.output(self.__trig_p, 0)
        time.sleep(0.0002)
        GPIO.output(self.__trig_p, 1)
        time.sleep(0.0001)
        GPIO.output(self.__trig_p, 0)
    
    def __wait_for_e(self, value: bool, timeout: int):
        '''
        必要的时延
        :param value
        :param timeout: 毫秒级时延
        :return:
        '''
        count = timeout
        while GPIO.input(self.__echo_p) != value and count > 0:
            count = count - 1
    
    def get_distance(self):
        '''
        通过传感器返回值进行计算
        :return: None
        '''
        self.__sent_t_pulse()
        self.__wait_for_e(True, 10000)
        start = time.time()
        self.__wait_for_e(False, 10000)
        finish = time.time()
        pulse_len = finish - start
        distance_cm = pulse_len / 0.000058
        distance_cm = round(distance_cm, 2)  # 保留两位小数  使用round内置函数
        self.dis = int(distance_cm)


# 普通io口的蜂鸣器,有源蜂鸣器
class Beep(object):
    '''
    Introduction:
        这是控制有源蜂鸣器的类
    Attributes:
    '''
    def __init__(self, beep_io: int):
        '''
        初始化 Beep 类
        :param beep_io:蜂鸣器的IO口，整数类型
        '''
        self.__gpio = IO2GPIO[beep_io]
        GPIO.setup(self.__gpio, GPIO.OUT)
        GPIO.output(self.__gpio, GPIO.HIGH)
    
    def beep_s(self, seconds: int):
        '''
        设置蜂鸣器响的时间
        :param seconds:蜂鸣器响的时间，整数类型
        :return:
        '''
        GPIO.output(self.__gpio, GPIO.LOW)
        time.sleep(seconds)
        GPIO.output(self.__gpio, GPIO.HIGH)
    
    def open_b(self):
        '''
        打开蜂鸣器
        :return: None
        '''
        GPIO.output(self.__gpio, GPIO.LOW)
    
    def close_b(self):
        '''
        关闭蜂鸣器
        :return: None
        '''
        GPIO.output(self.__gpio, GPIO.HIGH)


class Led(object):
    '''
    Introduction:
        这是控制Led灯亮灭的类
    Attributes:None

    Example:

    '''
    def __init__(self, led_io: int):
        '''
        初始化 Led 类
        :param led_io: led灯所调用的IO口，整数类型
        '''
        self.__gpio = IO2GPIO[led_io]
        GPIO.setup(self.__gpio, GPIO.OUT)
        GPIO.output(self.__gpio, GPIO.HIGH)
    
    def openled(self):
        '''
        打开led灯
        :return:None
        '''
        # 灯亮
        GPIO.output(self.__gpio, GPIO.HIGH)
    
    def closeled(self):
        '''
        关闭led灯
        :return:None
        '''
        GPIO.output(self.__gpio, GPIO.LOW)


class TempHump(object):
    '''
    Introduciton:
        这是用来控制温湿度传感器的类
    Attributes:
        temperature:若读取成功，读取到的温度值
        humidity:若读取成功，读取到的湿度值
        data: 读取成功与否的反馈值
    Example:

    '''
    def __init__(self, t_h_io: int):
        '''
        初始化TenpHump类
        :param t_h_io: 温湿度传感器的IO口，整数类型
        '''
        self.__gpio = IO2GPIO[t_h_io]
        GPIO.setup(self.__gpio, GPIO.IN)
        self.temperature = 'none'
        self.humidity = 'none'
        self.data = 'None'
    
    def getTemp_Humi(self):
        '''
        获取温湿度传感器返回的数值
        :return: None
        '''
        tmp = Adafruit_DHT.DHT11
        humi, temp = Adafruit_DHT.read_retry(tmp, self.__gpio)
        if temp == None or humi == None:
            self.data = "获取温湿度失败"
        else:
            self.data = "获取温度成功"
            self.temperature = str(temp) + '℃'
            self.humidity = str(humi) + '%'


class HongWai(object):
    '''
    Introduction:
        这是用来调用红外传感器的类
    Attributes:
        data:红外传感器的返回值，整数类型
        data_str:是否被遮挡，字符类型
    Example:
    '''
    # 红外检测模块
    def __init__(self, ir_io:int):
        '''
        初始化HongWai类
        :param ir_io:红外传感器调用的IO口，整数类型


        '''
        self.__gpio = IO2GPIO[ir_io]
        GPIO.setup(self.__gpio, GPIO.IN)
        # 红外IO设置为输入模式
        self.data = 0
        self.data_str = '无遮挡'
        # 没有东西遮挡为False
    
    def get_return(self):
        '''
        开始获取红外的返回值
        :return: None
        '''
        # 获取GPIO口的输入电平
        if GPIO.input(self.__gpio) == 0:
            # 低电平,返回false
            time.sleep(0.01)
            if GPIO.input(self.__gpio) == 0:
                # 防抖设计
                self.data = 0
                self.data_str = '无遮挡'
                # 输入为低电平
        else:
            self.data_str = '有遮挡'
            self.data = 1
            time.sleep(0.01)


class Servo():
    '''
    采用PCA9685芯片提供稳定的PWM波，弃用树莓派的模拟PWM波
    '''
    def __init__(self, pin:int):
        self.MIN_IMP = 500
        self.MAX_IMP = 2460
        self.servo_io = PWM2GPIO[pin]
        GPIO.cleanup(self.servo_io)  # 清空PWM口
        self.pca = ServoKit(channels=16)
        self.fa = 'False'  # 设置一个flag
        self.angle = None
        self.dic = {'True': '已经打开到', 'False': '当前角度为'}

    def init_servo(self):
        '''
        初始化舵机
        Returns:

        '''
        self.pca.servo[self.servo_io].set_pulse_width_range(self.MIN_IMP, self.MAX_IMP)  # 设置PWM口电平宽度
        self.pca.servo[self.servo_io].angle = 180
        time.sleep(0.01)
        self.angle = 0
        print("已完成舵机初始化")

    def turn(self, flage:str, angle1:int, delta:int):
        '''
        舵机旋转
        Args:
            flage: 模式
            angle1: 角度
            delta: 每次旋转的角度

        Returns:

        '''
        if 180 < angle1 or angle1 < 0:
            print("输入错误")
            return False
        curr_angle = self.angle
        diff_angle = abs(curr_angle - angle1)
        while diff_angle >= delta:
            # print("diffferent", diff_angle)
            # print(curr_angle)

            if flage == "open":
                if curr_angle > angle1:
                    print("输入角度小于之前角度！")
                    self.fa = 'False'
                    break
                else:
                    self.fa = 'True'
                    curr_angle += delta
            elif flage == "close":
                if curr_angle < angle1:
                    print("输入角度大于之前角度!")
                    self.fa = 'False'
                    break
                else:
                    self.fa = 'True'
                    curr_angle -= delta
            self.pca.servo[self.servo_io].angle = 180-curr_angle
            time.sleep(0.05)
            diff_angle = abs(curr_angle - angle1)
        self.angle = curr_angle
        if flage == "open":
            print(f"{self.dic[self.fa]}{self.angle}度")
        elif flage == "close":
            print(f"{self.dic[self.fa]}到{self.angle}度")


class Mecanum_wheel():
    '''
    初始化麦轮
    '''
    def __init__(self):
        self.ser = serial.Serial('/dev/ttyAMA0', 115200)
        self.dec = 'none'
        self.contr_fb = 0
        self.contr_lr = 0
        self.contr_tn = 0
        self.came = 0  # 识别到等待线之后才把cam置为1

        GPIO.setwarnings(False)  # 关闭警告说明
        GPIO.setup(0, GPIO.IN)  # 设置引脚1（BCM编号）为输入通道1GPIO.setup(0, GPIO.IN) #设置引脚1（BCM编号）为输入通道
        GPIO.setup(1, GPIO.IN)  # 设置引脚1（BCM编号）为输入通道

    def uart_init(self):
        '''
        初始化串口
        Returns:

        '''
        if self.ser.isOpen == False:
            self.ser.open()  # 打开串口

    def uart_receive(self):
        '''
        串口接收数据
        Returns:

        '''
        try:
            # 打开串口
            if self.ser.is_open == False:
                self.ser.open()
            while True:
                count = self.ser.inWaiting()
                if count != 0:
                    # 读取内容并显示
                    recv = self.ser.read(count)
                    print(recv)

                # 清空接收缓冲区
                self.ser.flushInput()
                # 必要的软件延时
                time.sleep(0.1)
        except KeyboardInterrupt:
            if self.ser != None:
                self.ser.close()

    '''
    (YYJ 2022/2/27)
    将之前两个块变成一个块，直接传递参数即可
    '''

    def car_stop(self):
        '''
        小车停止
        Returns:

        '''
        self.car_contr(0, 0, 0)

    def car_go(self, speed:int):
        '''
        小车前进
        Args:
            speed: 速度

        Returns:

        '''
        self.car_contr(speed, 0, 0)

    def car_back(self, speed:int):
        '''
               小车后退
               Args:
                   speed: 速度

               Returns:

               '''
        self.car_contr(-speed, 0, 0)

    def car_across_l(self, speed:int):
        '''
               小车左平移
               Args:
                   speed: 速度

               Returns:

               '''
        self.car_contr(0, -speed, 0)

    def car_across_r(self, speed:int):
        '''
        小车右平移
        Args:
            speed: 速度

        Returns:

        '''
        self.car_contr(0, speed, 0)

    '''
    (YYJ 2022/2/27)
    car_contr()第三个参数为 角度/s，下面是将线速度转换成 角度/s 过程:
    大致测量计算得到小车半径为 155mm ，测试得到浮点数会保留个位，正好取155(与速度同单位可以抵消）
    考虑到w单位是弧度制而这里是代码 角度/s 有个因子 57（1弧度约为57度）
    再考虑是四个个轮子造成差速，两个为一对，需要引入因子 4（这行不严谨半猜半想的，有待考究）
    实测得到 80度/s 是大概对应的90度/s  于是再引入修正因子 9/8
    推导原式： v = w*r
    得到的“修正式”：  W = (v/155)*(57/2)*8/9
    得到除以的因子约为 12   （带入电机速度对应的单片机参数对照差不多）
    注意：速度单位都是mm！！！
    '''

    def car_turn_l(self, speed:int):
        '''
               小车左转
               Args:
                   speed: 速度

               Returns:

            '''

        self.car_contr(0, 0, speed)

    def car_turn_r(self, speed:int):
        '''
               小车右转
               Args:
                   speed: 速度

               Returns:

        '''
        self.car_contr(0, 0, -speed)

    '''
    (YYJ 2022/3/1)
    这下面是平移运动函数。关于命名字母分别为
    L:Left; R:Right; F:forward; B:Back
    L_F：左 前 方行驶的意思，其他类似推可得意思。
    目前是给的两个参数一样，即45度角的平移
    '''

    def car_parallel_L_F(self, speed:int):
        '''
        左平移+前进
        Args:
            speed:

        Returns:

        '''
        self.car_contr(speed, -speed, 0)

    def car_parallel_R_F(self, speed:int):
        '''
        右平移+前进
        Args:
            speed:

        Returns:

        '''
        self.car_contr(speed, speed, 0)

    def car_parallel_L_B(self, speed:int):
        '''
        左平移+后退
        Args:
            speed:

        Returns:

        '''
        self.car_contr(-speed, -speed, 0)

    def car_parallel_R_B(self, speed:int):
        '''
        右平移+后退
        Args:
            speed:

        Returns:

        '''
        self.car_contr(-speed, speed, 0)

    '''
    (YYJ 2022/3/2)
    这下面是前进叠加旋转，类似于汽车进行的转弯运动
    根据一系列近似模型计算得到（这里记car_contr(x,y,w)方便阐述）
    x即为线速度，半径 R = (x/v)*d = 110*x/v   （这里v指的是旋转参数带来的线速度） 单位为mm
    所以得到 v = 110*x/R   
    用户输入线速度和半径就可求出旋转参数
    暂时写两个个函数表达四个方向（用户输入线速度正好有正负） —— 22/3/2
    参数与方向关系（用++表示 x为+ w为+，左传右转表示方向盘旋转较好理解）
    ++：向前左转  +-：向前右转  --：向后左转  -+：向后右转   F,B,L,R与上面一样意思
    如果半径单位也是mm没注意当成cm输入会造成半径很小，即旋转速度给到很大很大（单位待考究,目前是统一mm）
    '''

    def car_circle_L(self, speed:int, radius:int):  # 左传（后面不做解释了）
        '''
        左旋转+前后
        Args:
            speed:
            radius:

        Returns:

        '''
        w = 110 * speed / radius
        w = w / 12  # 转换为角速度
        # print(speed, w)  # 这行是看参数的，注释掉方便以后调试
        self.car_contr(speed, 0, w)

    def car_circle_R(self, speed:int, radius:int):
        '''
        右旋转+前后
        Args:
            speed:
            radius:

        Returns:

        '''
        w = 110 * speed / radius
        w = - w / 12  # 这两行一定要分开来写，不然数据帧会出现问题（符号问题看上面注释）
        self.car_contr(speed, 0, w)

    # 小车控制函数
    def car_contr(self, contr_fb:int=0, contr_lr:int=0, contr_tn:int=0):
        '''
        目前所用协议为 V1.0 ChenZuHong 2021-10-9
        :param contr_fb: 控制小车前进，协议中正数前进，负数后退，单位为mm/s
        :param contr_lr: 控制小车平移，协议中正数右平移，负数左平移，单位为mm/s
        :param contr_tn: 控制小车旋转，协议中正数逆时针，负数顺时针，单位为°/s
        '''
        global old_fb, old_lr, old_tn
        old_fb = contr_fb
        old_lr = contr_lr
        old_tn = contr_tn
        # 当速度为负的，做数据处理，得到电机反转的速度
        # fb 控制前后移动，lr控制左右平移，tn控制左右转向
        # fb = -10表示前进，lr = -10 表示向左平移，tn = -500表示左转
        if contr_fb < 0:
            contr_fb = 65536 + contr_fb
        if contr_lr < 0:
            contr_lr = 65536 + contr_lr
        if contr_tn < 0:
            contr_tn = 65536 + contr_tn
        byte_list = [0x55, 0x0E, 0x01, 0x01,
                     int(contr_fb / 256), int(contr_fb % 256),
                     int(contr_tn / 256), int(contr_tn % 256),
                     int(contr_lr / 256), int(contr_lr % 256),
                     0, 0, 1]
        k = 0
        for i in range(len(byte_list)):
            k += byte_list[i]
            k = k % 256
        byte_list.append(k)
        # 格式化要发送的数据帧
        contr_law = b"%c%c%c%c%c%c%c%c%c%c%c%c%c%c" % (byte_list[0], byte_list[1], byte_list[2], byte_list[3],
                                                       byte_list[4], byte_list[5], byte_list[6], byte_list[7],
                                                       byte_list[8], byte_list[9], byte_list[10], byte_list[11],
                                                       byte_list[12], byte_list[13])
        '''
        byte_list[0], byte_list[1], byte_list[2], byte_list[3]: 数据帧前四位， 协议中是 0x55, 0x0E, 0x01, 0x01
        byte_list[4], byte_list[5]: 协议中控制前进速度高八位、低八位
        byte_list[6], byte_list[7]: 协议中控制旋转速度高八位、低八位
        byte_list[8], byte_list[9]: 协议中控制平移速度高八位、低八位
        byte_list[10], byte_list[11]:保留位，默认为0，0
        byte_list[12], byte_list[13]:帧ID，默认为1， 校验位，由前面13个数据叠加而成
        '''
        # 发送数据帧
        self.ser.write(contr_law)

        # 防止连续快速发送数据导致出错
        time.sleep(0.005)

    def xunxian(self, io_l:int, io_r:int):  # 该函数是红外巡线，遇到白线跳出程序
        '''
        设置红外的io口
        Args:
            io_l:
            io_r:

        Returns:

        '''
        self.hw_l = HongWai(io_l)
        self.hw_r = HongWai(io_r)

        while True:
            # 获取返回数据
            self.hw_l.get_return()
            self.hw_r.get_return()
            self.came = 0
            if self.hw_l.data == 1 and self.hw_r.data == 1:
                self.car_contr(300, 0, 0)  # 三个参数分别代表/前进后退/左右平移/左右旋转的速度

            if self.hw_l.data == 0 and self.hw_r.data == 1:
                self.car_contr(300, 0, -30)

            if self.hw_l.data == 1 and self.hw_r.data == 0:
                self.car_contr(-300, 0, 30)

            if self.hw_l.data == 0 and self.hw_r.data == 0:  # 如果遇到白线（两个红外都返回0）跳出循环结束函数
                self.car_contr(0, 0, 0)
                self.came = 1
                break


class Screen():
    '''
    初始化LCD屏幕
    '''
    def __init__(self):
        pass

    def screen_display(self, string:str, background_color:str='white', font_color:str='black', font_size:int=20, Font:int=1, a:int=0, b:int=0):
        '''
        LCD屏幕显示
        Args:
            string:
            background_color:
            font_color:
            font_size:
            Font:
            a:
            b:

        Returns:

        '''
        # 屏幕大小为240*320，background_color为背景颜色，font_color字体颜色，font_size字体大小，a,b坐标

        # display with hardware SPI:
        ''' Warning!!!Don't  creation of multiple displayer objects!!! '''
        # disp = LCD_2inch4.LCD_2inch4(spi=SPI.SpiDev(bus, device),spi_freq=10000000,rst=RST,dc=DC,bl=BL)
        disp = LCD_2inch4.LCD_2inch4()
        disp.Init()
        disp.clear()

        image1 = Image.new("RGB", (320, 240), background_color)
        draw = ImageDraw.Draw(image1)

        logging.info("draw text")
        '''
        下面是字体的选择
        '''
        if Font == 1:
            Font = ImageFont.truetype(font_path + "Font00.ttf", font_size)
        elif Font == 2:
            Font = ImageFont.truetype(font_path + "Font01.ttf", font_size)
        elif Font == 3:
            Font = ImageFont.truetype(font_path + "Font02.ttf", font_size)
        elif Font == 4:
            Font = ImageFont.truetype(font_path + "Font03.ttf", font_size)

        font_number_max = int(320 / font_size - 1)
        # 字体一行最多的数量
        if len(string) < font_number_max:
            font_location = (a, b)
            draw.text(font_location, string, fill=font_color, font=Font)
        else:
            font_line_number = int(len(string) / font_number_max) + 1
            for i in range(font_number_max):
                font_location0 = (a, b + font_size * i)
                string0 = string[font_number_max * i:font_number_max * (i + 1)]
                draw.text(font_location0, string0, fill=font_color, font=Font)

        image1 = image1.rotate(0)
        disp.ShowImage(image1)

    def screen_display_picture(self, image_path:str):
        '''
        显示图片
        Args:
            image_path:

        Returns:

        '''
        disp = LCD_2inch4.LCD_2inch4()
        disp.Init()
        disp.clear()
        image = Image.open(image_path)
        image = image.resize((320, 240), Image.ANTIALIAS)

        disp.ShowImage(image)

    def video_show(self, video_path:str):
        '''

        Args:
            video_path: 视频文件位置

        Returns:

        '''

        cap = cv2.VideoCapture(video_path)
        disp = LCD_2inch4.LCD_2inch4()
        disp.Init()
        disp.clear()
        picture_show_time = cap.get(7)

        while cap.isOpened:
            ret, frame = cap.read()
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = image.resize((320, 240), Image.ANTIALIAS)
            #     image = image.filter(ImageFilter.SHARPEN)
            disp.ShowImage(image)
            picture_show_time -= 1
            cv2.waitKey(1)
            if not picture_show_time:
                break

        cap.release()

    def live_view_camera(self):
        '''
        实时显示摄像头得到的图像
        Returns:

        '''
        disp = LCD_2inch4.LCD_2inch4()
        disp.Init()
        disp.clear()
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            # cv2.imshow("capture", frame)
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = image.resize((320, 240), Image.ANTIALIAS)
            image = image.filter(ImageFilter.SHARPEN)
            disp.ShowImage(image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()



