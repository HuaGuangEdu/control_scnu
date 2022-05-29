#!/usr/local/bin/python3

"""
这是用来实现机器人语音的源码程序
"""
import warnings
warnings.filterwarnings("ignore")
from control.unique import  Number_Convert,playsound,Yuyin_local
from aip.speech import AipSpeech
import os, re,json,threading,subprocess
import wave
import time
import pyaudio
import audioop
import pygame
import requests
import urllib
import sys
import webbrowser
import random
from importlib import reload
# from unique import playsound
# os.close(sys.stderr.fileno())

system_platform = sys.platform

# 读取和保存文件所用主文件夹
main_path = '/home/pi/class/'
if 'win' in system_platform:
    # 获取当前文件的位置

    file_path = os.path.join(os.getcwd().split('blockly-electron')[0],'blockly-electron')
    if not os.path.exists(file_path):
        if os.path.exists(os.path.join(os.getcwd(),"resources")):
            file_path = os.getcwd()
    main_path = os.path.join(file_path , 'resources','assets','class').replace("\\","/")
# 文本文件夹
txt_path = os.path.join(main_path , 'txt/').replace("\\","/")

# 音频文件夹
audio_path = os.path.join(main_path , 'speech/').replace("\\","/")
if not os.path.exists(audio_path):
    os.makedirs(audio_path)
# # 开始时删除所有合成音频--Nonexxxxxxx.mp3/wav(固定格式)
for t in os.listdir(audio_path):
    if t.split(".")[-1] in ["mp3","wav"]:
        os.remove(audio_path+t)

# 百度API账号
app_id = '19925995'
app_key = '7GRa93EkYrOyFTfDkjHdl9WH'
app_secret_key = 'Q5qIyUFKP7U2ktBE4Y5oSUcom2x2v8sT'

# DUI平台提供的音色
ID = {"粤语女声何春": "hchunf_ctn", "男声小军": "xijunma", "知性女声晶晶": "jjingfp",
      "山东话女声大瑶": "dayaof_csd", "四川话女声胖胖": "ppangf_csn",
      "上海话女声叶子": "yezi1f_csh", "男声秋木": "qiumum_0gushi",
      "客服女声芳芳": "gdfanfp"}



# 初始类 Yuyin
class Yuyin():
    """
    Introduction:
        Yuyin类是用来为机器人提供一系列语音操作的，如播放语音，文字转语音，人机语音交互等
    Attributes:
        app_id:百度API登录ID
        app_key:在百度API中用户的标志
        app_secret_key:用户登录百度API的密码
        pyaudio.PyAudio():pyaudio库的PyAudio方法
        AipSpeech:百度语音API中的方法，是语音识别的Python SDK客户端提供语音识别一系列交互方法
    """
    def __init__(self,  **kwargs):
        """
        初始化Yuyin类
        :param None
        """
        self.online = True # 这个参数是针对本地化语音转文字的，如果是True就是调用百度在线的，否则调用本地化的
        for key,value in kwargs.items():
            if key=="online":
                self.online = value and ('win' in system_platform)
        if 'win' not in system_platform:
            print("树莓派上暂不支持本地模式，已为你切换成在线模式")
            self.online=True
        #下面这三个是写死的
        self.app_id = app_id
        self.api_key = app_key
        self.secret_key = app_secret_key

        self.p = pyaudio.PyAudio()

        self.client = AipSpeech(self.app_id, self.api_key, self.secret_key)

        # 语速和音量和音高（频率）--百度api
        self.vol = 9  # 感觉并没有什么特别大的差别
        self.spd = 5  # 数值一般为0-10
        self.per = 0  # 默认女声（数值只能0-5，但是除了男女声之外差别不大）

        # DUI
        self.vol_DUI = 100
        self.spd_DUI = 1
        self.gender = "xijunma"

        self.NumConverter = Number_Convert() #把百度的语音转文字中的中文数字转化成阿拉伯数字





    def change_vol_spd_gender_DUI(self, vol:int, spd:int, gender:str):
        """
        选择机器人播放时候的音量，播放速度以及声线（DUI版）
        网站：https://www.duiopen.com/docs/ct_cloud_TTS_Voice
        :param vol: 语音播放时候的音量
        :param spd: 语音播放时候的速度
        :param gender: 语音播放的声线
        :return: None
        """
        self.vol_DUI = vol
        self.spd_DUI = spd
        self.gender = ID.get(gender, None)

        # 手动抛出异常，防止输入错误的self.gender而导致程序崩溃
        if not self.gender:
            raise KeyError("没有这个音色！")

    def chat(self, my_text:str):
        """
        在百度API获取聊天机器人，将聊天机器人的语句通过self.chat_ret返回
        :param my_text: 对机器人说的话，以str类型输入
        :return: None
        """
        iner_url = 'http://api.qingyunke.com/api.php?key=free&appid=0&msg={}'
        url = iner_url.format(urllib.parse.quote(my_text))
        html = requests.get(url)
        self.chat_ret = html.json()["content"]

    def downsampleWav(self, src:str, dst:str, inrate:int=48000, outrate:int=16000, inchannels:int=1, outchannels:int=1):
        """
        修改成语音文件格式到适合百度语音api
        :param src: 原来的录音文件
        :param dst: 经过一系列操作后的录音文件
        :param inrate: 输入频率
        :param outrate: 输出频率
        :param inchannels: 输入通道
        :param outchannels: 输出通道
        :return: None
        """
        if not os.path.exists(src):
            print('没有旧音频文件')
            return False
        if not os.path.exists(os.path.dirname(dst)):
            os.makedirs(os.path.dirname(dst))

        # 打开WAV文件并获取其中的参数，参数以元组保存
        try:
            s_read = wave.open(src, 'rb')
            s_write = wave.open(dst, 'wb')
        except:
            print('打开旧音频文件失败')
            return False

        # 返回音频的帧数
        n_frames = s_read.getnframes()
        # 从流的当前指针位置一次读出音频的n个帧，并且指针后移n个帧，返回一个字节数组，给到data
        data = s_read.readframes(n_frames)

        try:
            # 转换输入片段的帧速率
            converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
            if outchannels == 1 and inchannels != 1:
                # 立体片段转换为单声道片段
                converted = audioop.tomono(converted[0], 2, 1, 0)
        except:
            print('转换格式失败')
            return False

        try:
            # 对WAV文件进行写入操作
            s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
            s_write.writeframes(converted[0])
        except Exception as e:
            print(e)
            print('保存新音频失败')
            return False

        try:
            s_read.close()
            s_write.close()
        except:
            print('无法关闭音频文件')
            return False
        return True


    def my_record(self, TIME:int, file_name:str):
        """
        机器人录音，并将录音保存到.wav文件
        :param TIME: 录音时间长度
        :param file_name: 录音file_name路径文件名
        :return: None
        """

        FORMAT = pyaudio.paInt16
        CHANNELS = 1  # 声道
        if self.online:
            CHUNK = 2000  # 采样点
            RATE = 48000  # 采样率
        # RECORD_SECONDS = 2                        # 采样宽度2bytes
        else:
            CHUNK = 1024
            RATE = 16000

        # 用时间戳和file_name作为文件名，时间戳保证文件的独特性
        try:
            file_name = audio_path + str(file_name) +  '.wav'
            stream = self.p.open(format=FORMAT,
                                 channels=CHANNELS,
                                 rate=RATE,
                                 input=True,
                                 frames_per_buffer=CHUNK)
            print("开始录音,请说话,持续", TIME, "秒......")
            frames = []
            t = time.time()
            while time.time() < t + TIME:
                data = stream.read(CHUNK)
                frames.append(data)
            print("录音结束!")
            # 停止音频流并关闭
            stream.stop_stream()
            stream.close()
        except:
            print("打开电脑音频失败，请重新尝试")
            exit()

        # 打开WAV文件，以二进制写模式，并对WAV文件进行一系列操作
        wf = wave.open(file_name, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        if self.online:
            file_new_name = audio_path + 'new.wav'
            # 通过downsampleWav（）函数对录音的音频文件进行修改
            self.downsampleWav(file_name, file_new_name)

            # 删除原来的录音文件
            os.remove(file_name)

            # 把file_name这个文件名给到修改后的文件
            os.rename(file_new_name, file_name)
        else:
            pass

    def stt(self, filename:str):
        """
        语音识别返回识别结果字符串, 识别.wav文件中的语音,  中文普通话识别的免费次数为50000次。
        :param filename: 要进行转换的文本文件
        :return: None
        """
        filename = audio_path + str(filename) + '.wav'
        if self.online:
            try:
                if os.path.exists(filename):

                    fp = open(filename, 'rb')
                    FilePath = fp.read()
                    fp.close()
                    # 识别本地文件
                    result = self.client.asr(FilePath,
                                             'wav',
                                             16000,
                                             {'dev_pid': 1537, }  # dev_pid参数表示识别的语言类型，1536表示普通话
                                             )

                    # 解析返回值，打印语音识别的结果
                    if result['err_msg'] == 'success.':
                        word = result['result'][0] # utf-8编码
                        numList = self.NumConverter.num_convert3(word)[1]
                        self.recordNumberList = [num[0] for num in numList]
                        return self.NumConverter.num_convert3(word)[0] # 返回识别结果值
                    else:
                        if 'win' not in system_platform:
                            addStr = "是不是没装麦克风？文件名是:"
                        else:
                            addStr = "文件名是:"
                        return "语音识别失败"+addStr + filename
            except:
                return "没有连接网络"

        else: #本地化语音转文字
            local_yuyinPath = os.path.join(audio_path, "local_yuyin") #本地化语音模型存放地点
            preWorkDir = os.getcwd() #将目前工作路径记录下来
            os.chdir(local_yuyinPath) #切换工作路径到本地化语音模型路径
            local_yuyin = Yuyin_local(record_time_s=-1,local_yuyinPath=local_yuyinPath,asyn=False,filename=filename)
            # local_yuyin.filename = filename
            local_yuyin.run()
            os.chdir(preWorkDir) #将工作路径切换回去
            return local_yuyin.total_sentance


    def tts(self, txt:str,filename:str):
        """
        将文本转为音频  语音合成免费额度只有5000次（未认证），认证之后有50000次，在180天内有效
        :param txt: 转语音的文本
        :param filename: 转换为音频的文件名
        :param tmp: 1使用百度api，2使用DUI，暂时使用，默认2
        :return: None
        """

        if not isinstance(txt,str):
            print("请输入字符串类型")

        if len(txt) != 0:
                try:
                    url = "https://dds.dui.ai/runtime/v1/synthesize?voiceId=" + self.gender + \
                          "&speed=" + str(self.spd_DUI) + \
                          "&volume=" + str(self.vol_DUI) + \
                          "&text=" + txt

                    r = requests.get(url)
                    result = r.content
                except:
                    raise BaseError('没有连接网络')

                filename = str(filename)
                file = audio_path + filename + '.mp3'

                if os.path.exists(file):
                    os.remove(file)
                with open(file, 'wb') as f:
                    f.write(result)

    def asyn_speech2text(self,record_time_s:int):
        '''
        一边说话一边识别
        :return:
        '''
        if self.online:
            self.my_record(record_time_s,"asy_yuyin")
            return self.stt("asy_yuyin")
        else:
            #下面涉及切换工作路径的原因可以参考上面stt的注释
            local_yuyinPath = os.path.join(audio_path, "local_yuyin")
            preWorkDir = os.getcwd()
            os.chdir(local_yuyinPath)
            local_yuyin = Yuyin_local(record_time_s,local_yuyinPath=local_yuyinPath)
            local_yuyin.run()
            os.chdir(preWorkDir)
            return local_yuyin.total_sentance


    def play_bufen(self, filename:str, play_time:int):
        """
        用于加载音频文件并播放
        :param filename: 音频文件
        :param play_time: 音频播放时间
        :return: None
        """
        pygame.mixer.init(frequency=16000, size=-16, channels=1, buffer=2000)
        track = pygame.mixer.music.load(audio_path + filename)
        pygame.mixer.music.play()
        time.sleep(play_time)
        pygame.mixer.music.stop()

    def play_music(self, filename:str):

        """
        播放音频及音乐,只能播放.mp3文件
        :param filename: 播放音频的文件名
        :param type: 默认为’。mp3‘
        :param model: 当 mode=1 的时候试播放音乐, 当 mode=0 的时候播放音频文件
        :param flag: 当 flag=0 的时候试全部播放, 当 flag=1 的时候试播放部分
        :param time: 音乐播放部分时候的播放时间
        :return: None
        """
        # filename = audio_path+str(filename)

        if os.path.exists (audio_path+filename+".mp3"):
            filename += ".mp3"
        elif os.path.exists (audio_path+filename+".wav"):
            filename += ".wav"
        else:
            raise FileNotFoundError("找不到该音频文件，是不是还没录制呢？")
        precwd = os.getcwd()
        os.chdir(audio_path)
        if 'win' in system_platform:
            playsound(filename)
        else:
            result = os.system("mplayer "+filename)
            if result != 0:
                print("这台树莓派上好像没有装mplayer(用于播放音频)，下面开始安装...")
                os.system("sudo apt install mplayer")
                os.system("mplayer "+filename)
        os.chdir(precwd)




    def play_txt(self, txt:str):
        '''
        将文本转换为语音并播放
        :param txt: 需要转换为音频的文本
        :return: None
        '''
        txt = str(txt)
        tmp = 'None'
        self.tts(txt, tmp)
        self.play_music(tmp)





