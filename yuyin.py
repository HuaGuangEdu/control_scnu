#!/usr/local/bin/python3

"""
这是用来实现机器人语音的源码程序
"""


from aip.speech import AipSpeech
import os, re
import wave
import time
import pyaudio
import audioop
import pygame
# import base64
import requests
import urllib
import sys
import webbrowser
import random
from importlib import reload
#os.close(sys.stderr.fileno())

system_platform = sys.platform

# 读取和保存文件所用主文件夹
main_path = '/home/pi/class/'
if 'win' in system_platform:
    # 获取当前文件的位置
    file_path = os.getcwd()
    main_path = file_path + '\\resources\\assets\\class\\'

# 文本文件夹
txt_path = main_path + 'txt\\'

# 音频文件夹
audio_path = main_path + 'speech\\'
if not os.path.exists(audio_path):
    os.makedirs(audio_path)
# # 开始时删除所有合成音频--Nonexxxxxxx.mp3/wav(固定格式)
# for i in os.listdir(audio_path):
#     t = i.split('.')
#
#     # 因为软件创建变量时默认定义为None，所以从第5个字符开始判断
#     if t[0][4:].isdigit():
#         os.remove(audio_path + i)

# 百度API账号
app_id = '19925995'
app_key = '7GRa93EkYrOyFTfDkjHdl9WH'
app_secret_key = 'Q5qIyUFKP7U2ktBE4Y5oSUcom2x2v8sT'

# DUI平台提供的音色
ID = {"粤语女声何春": "hchunf_ctn", "男声小军": "xijunma", "知性女声晶晶": "jjingfp",
      "山东话女声大瑶": "dayaof_csd", "四川话女声胖胖": "ppangf_csn",
      "上海话女声叶子": "yezi1f_csh", "男声秋木": "qiumum_0gushi",
      "客服女声芳芳": "gdfanfp"}


# 测试函数1
def test():
    print('hello lxy!')


# 测试函数2
def hello():
    print('you successful twice')


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

    def __init__(self, online=True):
        """
        初始化Yuyin类
        :param None
        """
        self.online = online #这个参数是针对本地化语音转文字的，如果是True就是调用百度在线的，否则调用本地化的

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

        self.NumConverter = Number_Convert()

    def change_vol_spd_gender(self, vol, spd, per):
        """
        选择机器人播放时候的音量，播放速度以及声线（百度版）
        :param vol: 语音播放时候的音量
        :param spd: 语音播放时候的速度
        :param per: 语音播放的声线,声线是使用百度API自带的
        :return: None
        """
        self.vol = 2 * vol - 1
        self.spd = 2 * spd - 1
        if per == 'young man':
            self.per = 1
        elif per == 'adult woman':
            self.per = 0
        elif per == 'adult man':
            self.per = 3
        elif per == 'young woman':
            self.per = 4

    def change_vol_spd_gender_DUI(self, vol, spd, gender):
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
            raise

    def chat(self, my_text):
        """
        在百度API获取聊天机器人，将聊天机器人的语句通过self.chat_ret返回
        :param my_text: 对机器人说的话，以str类型输入
        :return: None
        """
        iner_url = 'http://api.qingyunke.com/api.php?key=free&appid=0&msg={}'
        url = iner_url.format(urllib.parse.quote(my_text))
        html = requests.get(url)
        self.chat_ret = html.json()["content"]

    # def TxtRead(self, filename):
    #     '''
    #     读取文件并保存为字符串
    #     :param filename: 读取的文件名
    #     :return: 字符串类型的txt
    #     '''
    #     f = open(filename, "r")
    #     txt = f.read()
    #     f.close()
    #     return txt

    def downsampleWav(self, src, dst, inrate=48000, outrate=16000, inchannels=1, outchannels=1):
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
            params = s_read.getparams()
            nchannels, sampwidth, framerate, nframes = params[:4]
            # print(nchannels,sampwidth, framerate,nframes)
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


    def my_record(self, TIME, file_name):
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
        # p.terminate()

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

    def stt(self, filename):
        """
        语音识别返回识别结果字符串, 识别.wav文件中的语音,  中文普通话识别的免费次数为50000次。
        :param filename: 要进行转换的文本文件
        :return: None
        """
        if self.online:
            try:
                filename = audio_path + str(filename) + '.wav'
                fp = open(filename, 'rb')
                FilePath = fp.read()
                fp.close()
            except:
                print(filename + "音频文件不存在或格式错误")
            finally:
                try:
                    # 识别本地文件
                    result = self.client.asr(FilePath,
                                             'wav',
                                             16000,
                                             {'dev_pid': 1537, }     # dev_pid参数表示识别的语言类型，1536表示普通话
                                             )

                    # 解析返回值，打印语音识别的结果
                    if result['err_msg'] == 'success.':
                        word = result['result'][0]                    # utf-8编码
                        numList = self.NumConverter.num_convert3(word)[1]
                        self.recordNumberList = [num[0] for num in numList]
                        return self.NumConverter.num_convert3(word)[0]                                   # 返回识别结果值

                    else:
                        print("语音识别失败:" + filename)
                        return "语音识别失败"
                except:
                    print("没有连接网络")
                    return "没有连接网络"
        else:

            res = os.popen(f"{audio_path}decoder_main.exe  --chunk_size -1  --wav_path {audio_path+'None'+'.wav'}  \
                    --model_path {audio_path}final.zip --dict_path {audio_path}words.txt")

            tempstream = res._stream
            # return tempstream.buffer.read().decode(encoding='utf-8', errors='ignore').split(' ')[1].split('\r\n')[0]
            numList = self.NumConverter.num_convert3(tempstream.buffer.read().decode(encoding='utf-8', errors='ignore').split(' ')[1].split('\r\n')[0])
            word = [num[0] for num in numList]
            self.recordNumberList = word[1]
            return word[0]




    def tts(self, txt, filename, tmp=2):
        """
        将文本转为音频  语音合成免费额度只有5000次（未认证），认证之后有50000次，在180天内有效
        :param txt: 转语音的文本
        :param filename: 转换为音频的文件名
        :param tmp: 1使用百度api，2使用DUI，暂时使用，默认2
        :return: None
        """

        txt = str(txt)
        # 用时间戳和file_name作为文件名，时间戳保证文件的独特性
        if len(txt) != 0:
            if tmp == 1:
                word = txt
                # try:

                # synthesis（）用于语音合成
                result = self.client.synthesis(word, 'zh', 1, {
                    'vol': self.vol,  # 音量
                    'per': self.per,  # 音色--0 1 3 4（2和1差不多）
                    'spd': self.spd,  # 语速
                    'plt': 10  # 语调
                })

                # 合成正确返回audio.mp3，错误则返回dict
                if not isinstance(result, dict):
                    with open(audio_path + str(filename) + '.mp3', 'wb') as f:
                        f.write(result)
                        print('文字转音频成功:' + txt)
                else:
                    print('文字转音频失败!')
            elif tmp == 2:
                url = "https://dds.dui.ai/runtime/v1/synthesize?voiceId=" + self.gender + \
                      "&speed=" + str(self.spd_DUI) + \
                      "&volume=" + str(self.vol_DUI) + \
                      "&text=" + txt

                r = requests.get(url)
                result = r.content

                filename = str(filename) + '.mp3'
                file = audio_path + filename
                try:
                    with open(file, 'wb') as f:
                        f.write(result)
                except:
                    file = file+'1'
                    with open(file, 'wb') as f:
                        f.write(result)
    #         except:
    #             print('没有连接网络')

    def play_bufen(self, filename, play_time):
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

    def play_music(self, filename, type='.mp3', model=1, flag=0, time=0):
        """
        播放音频及音乐,只能播放.mp3文件
        :param filename: 播放音频的文件名
        :param type: 默认为’。mp3‘
        :param model: 当 mode=1 的时候试播放音乐, 当 mode=0 的时候播放音频文件
        :param flag: 当 flag=0 的时候试全部播放, 当 flag=1 的时候试播放部分
        :param time: 音乐播放部分时候的播放时间
        :return: None
        """
        pygame.mixer.init(frequency=16000, size=-16, channels=1, buffer=2000)
        filename = str(filename)  + type
        if type == '.wav':
            track = pygame.mixer.Sound(audio_path + filename)
            track.play()

        elif type == '.mp3':
            if model == 0:
                track = pygame.mixer.music.load(audio_path + filename)
                pygame.mixer.music.play()

            elif model == 1:
                track = pygame.mixer.music.load(audio_path + filename)
                if flag == 0:
                    pygame.mixer.music.play()

                    # 等待播放完毕
                    while pygame.mixer.music.get_busy():
                        if pygame.mixer.music.get_busy() == 0:
                            break

                else:
                    self.play_bufen(filename, time)

    def play_txt(self, txt):
        '''
        将文本转换为语音并播放
        :param txt: 需要转换为音频的文本
        :return: None
        '''
        txt = str(txt)
        tmp = None
        self.tts(txt, tmp)
        self.play_music(tmp)



#用来转换数字
class Number_Convert():
    def __init__(self):

        self.number_map = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8,
                           '九': 9}  # 1-9数字
        self.unit_map = {'十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}  # 数字单位


    def __operate(self, num_str):  # 这个和下面呢个____operate1都是处理字符串的函数，别调用
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

    def __operate1(self, strings):  # 处理字符串的，分成了三种情况，有“亿”，无“亿”有“万”， 无“亿”无“万”

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

    def num_convert3(self, test_strings):
        self.NumList = [] #装数字的列表
        self.converted_strings = ''  # 转化后的字符串
        self.test_strings = test_strings
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
        return [self.converted_strings, sorted(self.NumList, key=lambda x: x[1])]


# 测试录音+语音识别
# '''
# s=Yuyin()
# s.my_record(3,"1")
# txt=s.stt("1")
# print(txt)
# '''
# '''
# #测试文本转语音
# s=Yuyin()
# s.tts('The dog is eating shits!',"c4")  #tts保存为mp3格式
# s.play_music("c4.mp3")
# '''
# '''
# s=Yuyin()
# s.play_music("Build a temporary bridge.mp3")
# '''