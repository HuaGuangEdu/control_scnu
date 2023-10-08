#!/usr/local/bin/python3

"""
这是用来实现机器人语音的源码程序
"""
import json
import warnings

warnings.filterwarnings("ignore")
from .util.number_convert import Number_Convert
from .util.local_yuyin import Yuyin_local, Yuyin_local2
from .util.playsound_change import playsound
import os
import wave
import pyaudio
import requests
import urllib
import time
from .util.all_path import system_platform, class_path
from .util.download import download, getFileSize, models
import shutil

# 音频文件夹
audio_path = os.path.join(class_path, 'speech').replace("\\", "/")
if not os.path.exists(audio_path):
    os.mkdir(audio_path)
# 在运行程序之前先把原本有的音频文件全部删掉
for file in os.listdir(audio_path):
    if file.split(".")[-1] in ["mp3", "wav"]:
        os.remove(os.path.join(audio_path, file))

# 百度API账号
app_id = '19925995'
app_key = '7GRa93EkYrOyFTfDkjHdl9WH'
app_secret_key = 'Q5qIyUFKP7U2ktBE4Y5oSUcom2x2v8sT'

# DUI平台提供的音色
ID = {"粤语女声何春": "hchunf_ctn", "男声小军": "xijunma", "知性女声晶晶": "jjingfp",
      "山东话女声大瑶": "dayaof_csd", "四川话女声胖胖": "ppangf_csn",
      "上海话女声叶子": "yezi1f_csh", "男声秋木": "qiumum_0gushi",
      "客服女声芳芳": "gdfanfp"}
p = pyaudio.PyAudio()


def play_music(filename: str):
    """
    播放音频
    :param filename: 播放音频的文件名
    :return: None
    """

    if os.path.exists(os.path.join(audio_path, filename + ".mp3")):
        filename += ".mp3"
    elif os.path.exists(os.path.join(audio_path, filename + ".wav")):
        filename += ".wav"
    else:
        raise FileNotFoundError("找不到该音频文件，是不是还没录制呢？")
    precwd = os.getcwd()
    # 因为playsound输入的音频文件只能支持相对路径，你不能用绝对路径，所以要先把工作路径改到音频文件目录下然后播放音频之后再改回去
    os.chdir(audio_path)
    if 'win' in system_platform:
        # 如果在windows上，就用playsound来播放就行
        playsound(filename)
    else:
        # 如果在树莓派，就要用mplayer来播放
        result = os.system("mplayer " + filename)
        if result != 0:
            print("这台树莓派上好像没有装mplayer(用于播放音频)，下面开始安装...")
            os.system("sudo apt install mplayer")
            os.system("mplayer " + filename)
    os.chdir(precwd)


def get_music_file_return(filename: str):
    """
    获取对应路径的音频文件的字节流数据
    :param filename: 音频文件名
    :return: 音频文件字节流
    """
    if os.path.exists(os.path.join(audio_path, filename + ".mp3")):
        filename += ".mp3"
    elif os.path.exists(os.path.join(audio_path, filename + ".wav")):
        filename += ".wav"
    else:
        raise FileNotFoundError("找不到该音频文件，是不是还没录制呢？")
    with open(os.path.join(audio_path, filename), "rb") as f:
        return f.read()


def my_record(TIME: int, file_name: str):
    """
    机器人录音，并将录音保存到.wav文件
    :param TIME: 录音时间长度
    :param file_name: 录音file_name路径文件名，如speech，不需要后缀
    :return: None
    """

    FORMAT = pyaudio.paInt16
    CHANNELS = 1  # 声道
    CHUNK = 1024  # 采样点
    RATE = 16000  # 采样率

    # 用时间戳和file_name作为文件名，时间戳保证文件的独特性
    try:
        file_name = os.path.join(audio_path, str(file_name) + '.wav')
        stream = p.open(format=FORMAT,
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
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def my_record_and_return(TIME: int, file_name: str):
    my_record(TIME, file_name)
    return {"type": "audio", "data": get_music_file_return(file_name)}


def write_wav_bytes_to_file(filename: str, data):
    """
    将音频文件写入到文件中
    :param filename: 文件路径
    :param data: 文件内容
    :return:
    """
    if data:
        filename = os.path.join(audio_path, str(filename) + '.wav')
        with open(filename, "wb") as f:
            f.write(data)
    else:
        raise ValueError("data 不能为空")


class SpeechRecognizer:
    def __init__(self, model="online"):
        """
        语音识别，也就是把语音转换成文字
        :param model: 有四个选项online, k2_rnnt, conformer, stream如果是online就是调用百度在线，其他都是离线语音识别
        """
        if model not in ['online', 'k2_rnnt', 'conformer', 'stream']:
            raise ValueError("没有这个模型")
        self.model = model
        if model == 'online':
            from aip.speech import AipSpeech
            self.client = AipSpeech(app_id, app_key, app_secret_key)
            self.NumConverter = Number_Convert()  # 把百度的语音转文字中的中文数字转化成阿拉伯数字
        else:
            self.__checkFileExists(modelName=model)

    def __checkFileExists(self, modelName):
        """
        第一步先检查检查离线语音识别所需要的可执行文件是否存在在电脑上，如果没有的话就要从飞书上下载
        第二步再检查电脑里面是否有选择的模型，如果没有的话就要从飞书上下载
        :return: None
        """
        # 第一步
        checkFile = ("win" if "win" in system_platform else "linux") + "语音识别可执行文件"
        if os.path.exists(os.path.join(audio_path, "exeFile")) is False or getFileSize(
                os.path.join(audio_path, "exeFile")) != models[checkFile]["actual_size"]:
            try:
                shutil.rmtree(os.path.join(audio_path, "exeFile"))
            except:
                pass
            # 没有本地化语音的语音识别可执行文件，所以要下载
            print("未找到离线语音识别可执行文件，准备下载")
            download(checkFile)

        # 第二步
        if os.path.exists(os.path.join(audio_path, modelName)) is False or getFileSize(
                os.path.join(audio_path, modelName)) != models[modelName]["actual_size"]:
            # 没有找到本地化语音模型，准备下载模型
            print("没有找到本地化语音模型，准备下载模型")
            download(modelName)

    def stt(self, filename: str):
        """
        语音识别返回识别结果字符串, 识别.wav文件中的语音,
        在线语音识别调用百度api，中文普通话识别的免费次数为50000次。
        离线语音识别可选非流式k2_rnnt或conformer或流式的stream模型
        其中流式模型虽然是流式，但目前只能输入wav文件，不能一边录音一边识别，如果有兴趣的可以去飞书上面修改C++源码然后重新编译出可执行文件
        :param filename: 要进行转换的文本文件
        :return: None
        """
        filename = os.path.join(audio_path, str(filename) + '.wav')
        if self.model == 'online':
            try:
                if os.path.exists(filename):
                    fp = open(filename, 'rb')
                    FilePath = fp.read()
                    fp.close()
                    # 识别本地文件
                    result = self.client.asr(FilePath, 'wav', 16000, {'dev_pid': 1537})  # dev_pid参数表示识别的语言类型，1536表示普通话
                    # 解析返回值，打印语音识别的结果
                    if result['err_msg'] == 'success.':
                        word = result['result'][0]  # utf-8编码
                        # numList = self.NumConverter.num_convert3(word)[1]
                        # self.recordNumberList = [num[0] for num in numList] #把识别到的数字放到一个列表，暂时没用所以注释了，看看以后是否有用
                        return self.NumConverter.num_convert3(word)[0]  # 返回识别结果值
                    else:
                        if 'win' not in system_platform:
                            addStr = "是不是没装麦克风？文件名是:"
                        else:
                            addStr = "文件名是:"
                        return "语音识别失败" + addStr + filename
            except:
                return "没有连接网络"

        else:  # 本地化语音转文字
            local_yuyin2 = Yuyin_local2(modelName=self.model, wavFile=os.path.join(audio_path, filename))
            result = local_yuyin2.recognize()
            return result


class SpeechSynthesis:
    def __init__(self, online=True):
        """
        语音合成，就是把文字转换成语音
        :param online: 有两个选项，True就是调用百度在线语音合成，如果是False就是调用
        """
        self.online = online
        if online:
            # 调用百度在线合成
            from aip.speech import AipSpeech
            self.client = AipSpeech(app_id, app_key, app_secret_key)

            # 语速和音量和音高（频率）--百度api
            self.vol = 9  # 感觉并没有什么特别大的差别
            self.spd = 5  # 数值一般为0-10
            self.per = 0  # 默认女声（数值只能0-5，但是除了男女声之外差别不大）

            # DUI
            self.vol_DUI = 100
            self.spd_DUI = 1
            self.gender = "xijunma"
        elif "win" in system_platform:
            # 调用离线语音合成，但是在windows上，所以使用pyttsx3
            import pyttsx3
            self.engine = pyttsx3.init()
        else:
            # 调用离线语音合成，但是在树莓派上，所以使用zhtts
            import zhtts
            import soundfile as sf
            import sounddevice as sd
            self.sf = sf
            self.sd = sd
            self.rpiT2S = zhtts.TTS()

    def change_vol_spd_gender_DUI(self, vol: int, spd: int, gender: str):
        """
        选择机器人播放时候的音量，播放速度以及声线（DUI版）
        网站：https://www.duiopen.com/docs/ct_cloud_TTS_Voice
        :param vol: 语音播放时候的音量
        :param spd: 语音播放时候的速度
        :param gender: 语音播放的声线
        :return: None
        """
        if self.online:
            # 调用百度
            self.vol_DUI = vol
            self.spd_DUI = spd
            self.gender = ID.get(gender, None)

            if not self.gender:
                raise KeyError("没有这个音色！")
        elif "win" in system_platform:
            # 离线语音合成且在windows
            warnings.warn("成功设置语速的音量，windows离线语音合成不支持设置声线", UserWarning)
            self.engine.setProperty('rate', int(200 * (1 / spd)))  # 设置语速
            self.engine.setProperty('volume', 0.6 * 0.01 * vol)  # 设置音量

        else:
            # 离线语音合成且在树莓派
            warnings.warn("设置无效，树莓派中的离线语音合成无法设置音量、语速以及声线", UserWarning)

    def tts(self, txt: str, filename: str):
        """
        将文本转为音频  语音合成免费额度只有5000次（未认证），认证之后有50000次，在180天内有效
        :param txt: 转语音的文本
        :param filename: 转换为音频的文件名
        :return: None
        """
        if len(txt) == 0:
            raise ValueError("文字转语音中输入的字符串不能为空")
        filename = str(filename)

        if self.online:
            try:
                url = "https://dds.dui.ai/runtime/v1/synthesize?voiceId=" + self.gender + \
                      "&speed=" + str(self.spd_DUI) + \
                      "&volume=" + str(self.vol_DUI) + \
                      "&text=" + txt

                r = requests.get(url)
                result = r.content
            except:
                raise FileNotFoundError('没有连接网络')
            file = os.path.join(audio_path, filename + '.mp3')
            if os.path.exists(file):
                os.remove(file)
            with open(file, 'wb') as f:
                f.write(result)
        else:
            # pyttsx3或zhtts的音频都必须要保存成wav格式，不然playsound无法播放
            file = os.path.join(audio_path, filename + '.wav')
            if "win" in system_platform:
                # 离线语音合成且在windows
                if os.path.exists(file):
                    os.remove(file)
                self.engine.save_to_file(txt, file)
                self.engine.runAndWait()
            else:
                # 离线语音合成且在树莓派
                mel = self.rpiT2S.text2mel(txt)
                audio = self.rpiT2S.mel2audio(mel)
                self.sf.write(file, audio, 24000, 'PCM_16')

    def play_txt(self, txt: str):
        """
        将文本转换为语音并播放
        :param txt: 需要转换为音频的文本
        :return: None
        """
        if self.online:
            # 调用百度在线语音合成
            txt = str(txt)
            tmp = 'audio'
            self.tts(txt, tmp)
            play_music(tmp)
        elif "win" in system_platform:
            # 调用离线语音合成，但是在windows上，使用pyttsx3
            self.engine.say(txt)
            self.engine.runAndWait()
        else:
            # 调用离线语音合成，但是在树莓派上，使用zhtts
            mel = self.rpiT2S.text2mel(txt)
            audio = self.rpiT2S.mel2audio(mel)
            self.sd.play(audio, samplerate=24000)
            self.sd.wait()


class ChatBot:
    def __init__(self):
        """
        聊天机器人，目前只有在线调用百度的，以后可能有离线的聊天机器人
        """
        pass

    def chat(self, my_text: str):
        """
        在百度API获取聊天机器人，将聊天机器人的语句通过self.chat_ret返回
        :param my_text: 对机器人说的话，以str类型输入
        :return: None
        """
        iner_url = 'http://api.qingyunke.com/api.php?key=free&appid=0&msg={}'
        url = iner_url.format(urllib.parse.quote(my_text))
        html = requests.get(url)
        self.chat_ret = html.json()["content"]


class Yuyin:
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

    def __init__(self, **kwargs):
        """
        初始化Yuyin类
        :param None
        """
        print("这个类在未来将被废弃")
        warnings.warn("这个类在未来将被废弃", FutureWarning)
        self.online = True  # 这个参数是针对本地化语音转文字的，如果是True就是调用百度在线的，否则调用本地化的
        for key, value in kwargs.items():
            if key == "online":
                self.online = value
        # 下面这三个是写死的
        self.app_id = app_id
        self.api_key = app_key
        self.secret_key = app_secret_key

        self.p = pyaudio.PyAudio()
        from aip.speech import AipSpeech
        self.client = AipSpeech(self.app_id, self.api_key, self.secret_key)

        # 语速和音量和音高（频率）--百度api
        self.vol = 9  # 感觉并没有什么特别大的差别
        self.spd = 5  # 数值一般为0-10
        self.per = 0  # 默认女声（数值只能0-5，但是除了男女声之外差别不大）

        # DUI
        self.vol_DUI = 100
        self.spd_DUI = 1
        self.gender = "xijunma"

        self.NumConverter = Number_Convert()  # 把百度的语音转文字中的中文数字转化成阿拉伯数字
        if self.online is False:
            if "win" in system_platform:
                import pyttsx3
                self.engine = pyttsx3.init()
                if os.path.exists(os.path.join(audio_path, "local_yuyin")) is False or getFileSize(
                        os.path.join(audio_path, "local_yuyin")) != models['本地化语音']["actual_size"]:
                    # 没有本地化语音的模型，所以要下载模型
                    print("未发现模型或模型不完整，准备下载模型")
                    download("本地化语音")
            else:
                import zhtts
                import soundfile as sf
                import sounddevice as sd
                self.sf = sf
                self.sd = sd
                self.rpiT2S = zhtts.TTS()

    def change_vol_spd_gender_DUI(self, vol: int, spd: int, gender: str):
        """
        选择机器人播放时候的音量，播放速度以及声线（DUI版）
        网站：https://www.duiopen.com/docs/ct_cloud_TTS_Voice
        :param vol: 语音播放时候的音量
        :param spd: 语音播放时候的速度
        :param gender: 语音播放的声线
        :return: None
        """
        if self.online or "win" not in system_platform:
            self.vol_DUI = vol
            self.spd_DUI = spd
            self.gender = ID.get(gender, None)

            if not self.gender:
                raise KeyError("没有这个音色！")
        else:
            print("成功设置语速的音量，但本地文字转语音不支持设置声线")
            self.engine.setProperty('rate', int(200 * (1 / spd)))  # 设置语速
            self.engine.setProperty('volume', 0.6 * 0.01 * vol)  # 设置音量

    def chat(self, my_text: str):
        """
        在百度API获取聊天机器人，将聊天机器人的语句通过self.chat_ret返回
        :param my_text: 对机器人说的话，以str类型输入
        :return: None
        """
        iner_url = 'http://api.qingyunke.com/api.php?key=free&appid=0&msg={}'
        url = iner_url.format(urllib.parse.quote(my_text))
        html = requests.get(url)
        self.chat_ret = html.json()["content"]

    def downsampleWav(self, src: str, dst: str, inrate: int = 48000, outrate: int = 16000, inchannels: int = 1,
                      outchannels: int = 1):
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
        import audioop
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

    def my_record(self, TIME: int, file_name: str):
        """
        机器人录音，并将录音保存到.wav文件
        :param TIME: 录音时间长度
        :param file_name: 录音file_name路径文件名
        :return: None
        """

        FORMAT = pyaudio.paInt16
        CHANNELS = 1  # 声道
        if self.online or "win" not in system_platform:
            CHUNK = 2000  # 采样点
            RATE = 48000  # 采样率
        # RECORD_SECONDS = 2                        # 采样宽度2bytes
        else:
            CHUNK = 1024
            RATE = 16000

        # 用时间戳和file_name作为文件名，时间戳保证文件的独特性
        try:
            file_name = os.path.join(audio_path, str(file_name) + '.wav')
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
        if self.online or "win" not in system_platform:
            file_new_name = os.path.join(audio_path, 'new.wav')
            # 通过downsampleWav（）函数对录音的音频文件进行修改
            self.downsampleWav(file_name, file_new_name)

            # 删除原来的录音文件
            os.remove(file_name)

            # 把file_name这个文件名给到修改后的文件
            os.rename(file_new_name, file_name)
        else:
            pass

    def stt(self, filename: str):
        """
        语音识别返回识别结果字符串, 识别.wav文件中的语音,  中文普通话识别的免费次数为50000次。
        :param filename: 要进行转换的文本文件
        :return: None
        """
        filename = os.path.join(audio_path, str(filename) + '.wav')
        if self.online or "win" not in system_platform:
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
                        word = result['result'][0]  # utf-8编码
                        numList = self.NumConverter.num_convert3(word)[1]
                        self.recordNumberList = [num[0] for num in numList]
                        return self.NumConverter.num_convert3(word)[0]  # 返回识别结果值
                    else:
                        if 'win' not in system_platform:
                            addStr = "是不是没装麦克风？文件名是:"
                        else:
                            addStr = "文件名是:"
                        return "语音识别失败" + addStr + filename
            except:
                return "没有连接网络"

        else:  # 本地化语音转文字
            local_yuyinPath = os.path.join(audio_path, "local_yuyin")  # 本地化语音模型存放地点
            preWorkDir = os.getcwd()  # 将目前工作路径记录下来
            os.chdir(local_yuyinPath)  # 切换工作路径到本地化语音模型路径
            local_yuyin = Yuyin_local(record_time_s=-1, local_yuyinPath=local_yuyinPath, asyn=False, filename=filename)
            isError = local_yuyin.run()
            if isError:
                os.system("cls")
                local_yuyin.run()
            os.chdir(preWorkDir)  # 将工作路径切换回去
            return local_yuyin.total_sentance

    def tts(self, txt: str, filename: str):
        """
        将文本转为音频  语音合成免费额度只有5000次（未认证），认证之后有50000次，在180天内有效
        :param txt: 转语音的文本
        :param filename: 转换为音频的文件名
        :param tmp: 1使用百度api，2使用DUI，暂时使用，默认2
        :return: None
        """
        if len(txt) == 0:
            raise ValueError("文字转语音中输入的字符串不能为空")
        filename = str(filename)

        if self.online:
            try:
                url = "https://dds.dui.ai/runtime/v1/synthesize?voiceId=" + self.gender + \
                      "&speed=" + str(self.spd_DUI) + \
                      "&volume=" + str(self.vol_DUI) + \
                      "&text=" + txt

                r = requests.get(url)
                result = r.content
            except:
                raise FileNotFoundError('没有连接网络')
            file = os.path.join(audio_path, filename + '.mp3')
            if os.path.exists(file):
                os.remove(file)
            with open(file, 'wb') as f:
                f.write(result)
        else:
            # pyttsx3或zhtts的音频都必须要保存成wav格式，不然playsound无法播放
            file = os.path.join(audio_path, filename + '.wav')
            if "win" in system_platform:
                if os.path.exists(file):
                    os.remove(file)
                self.engine.save_to_file(txt, file.replace(".mp3", ".wav"))
                self.engine.runAndWait()
            else:
                mel = self.rpiT2S.text2mel(txt)
                audio = self.rpiT2S.mel2audio(mel)
                self.sf.write("a.wav", audio, 24000, 'PCM_16')

    def asyn_speech2text(self, record_time_s: int):
        """
        一边说话一边识别
        :return:
        """
        if self.online or "win" not in system_platform:
            self.my_record(record_time_s, "asy_yuyin")
            return self.stt("asy_yuyin")
        else:
            # 下面涉及切换工作路径的原因可以参考上面stt的注释
            local_yuyinPath = os.path.join(audio_path, "local_yuyin")
            preWorkDir = os.getcwd()
            os.chdir(local_yuyinPath)
            local_yuyin = Yuyin_local(record_time_s, local_yuyinPath=local_yuyinPath)
            isError = local_yuyin.run()
            if isError:
                os.system("cls")
                local_yuyin.run()
            os.chdir(preWorkDir)
            return local_yuyin.total_sentance

    def play_music(self, filename: str):

        """
        播放音频及音乐,只能播放.mp3文件
        :param filename: 播放音频的文件名
        :param type: 默认为’。mp3‘
        :param model: 当 mode=1 的时候试播放音乐, 当 mode=0 的时候播放音频文件
        :param flag: 当 flag=0 的时候试全部播放, 当 flag=1 的时候试播放部分
        :param time: 音乐播放部分时候的播放时间
        :return: None
        """

        if os.path.exists(os.path.join(audio_path, filename + ".mp3")):
            filename += ".mp3"
        elif os.path.exists(os.path.join(audio_path, filename + ".wav")):
            filename += ".wav"
        else:
            raise FileNotFoundError("找不到该音频文件，是不是还没录制呢？")
        precwd = os.getcwd()
        os.chdir(audio_path)
        if 'win' in system_platform:
            playsound(filename)
        else:
            result = os.system("mplayer " + filename)
            if result != 0:
                print("这台树莓派上好像没有装mplayer(用于播放音频)，下面开始安装...")
                os.system("sudo apt install mplayer -y")
                os.system("mplayer " + filename)
        os.chdir(precwd)

    def play_txt(self, txt: str):
        """
        将文本转换为语音并播放
        :param txt: 需要转换为音频的文本
        :return: None
        """
        if self.online:
            txt = str(txt)
            tmp = 'audio'
            self.tts(txt, tmp)
            self.play_music(tmp)
        else:
            if "win" in system_platform:
                self.engine.say(txt)
                self.engine.runAndWait()
            else:
                mel = self.rpiT2S.text2mel(txt)
                audio = self.rpiT2S.mel2audio(mel)
                self.sd.play(audio, samplerate=24000)
                self.sd.wait()
