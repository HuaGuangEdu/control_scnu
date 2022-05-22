import websocket,json,threading,os,pyaudio,subprocess,time,re

#用来转换数字
class Number_Convert():
    def __init__(self):

        self.number_map = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8,
                           '九': 9}  # 1-9数字
        self.unit_map = {'十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}  # 数字单位


    def __operate(self, num_str:str):  # 这个和下面呢个____operate1都是处理字符串的函数，别调用
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

    def __operate1(self, strings:str):  # 处理字符串的，分成了三种情况，有“亿”，无“亿”有“万”， 无“亿”无“万”

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

    def num_convert3(self, test_strings:str):
        self.NumList = [] #装数字的列表
        self.converted_strings = ''  # 转化后的字符串
        self.test_strings = test_strings.replace("什","【·&……】")
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
        return [self.converted_strings.replace("【·&……】","什"), sorted(self.NumList, key=lambda x: x[1])]

######################本地化语音########################

class Yuyin_local():
    def __init__(self,record_time_s,local_yuyinPath,asyn=True,filename=''):
        os.system("chcp 65001") #切换语音为Unicode (UTF-8)
        vbsPath = os.path.join(local_yuyinPath,"runbat.vbs")
        subprocess.call(f"cscript  {vbsPath}", stdout=None, stdin=None)
        self.ws_app = websocket.WebSocketApp("ws://127.0.0.1:10086",
                                        on_open=lambda ws: self.on_open(ws, record_time_s),  # 连接建立后的回调
                                        on_message=self.on_message,  # 接收消息的回调
                                        )
        self.total_sentance = '' #存放语音识别的内容
        self.asyn = asyn
        self.filename = filename
        p = pyaudio.PyAudio()
        self.stream = p.open(format=pyaudio.paInt16,
                        channels= 1,
                        rate=16000,
                        input=True,
                        frames_per_buffer=1024,
                        )

    def run(self):
        self.ws_app.run_forever()



    def on_message(self,ws, message):
        """
        接收服务端返回的消息
        :param ws:
        :param message: json格式，自行解析
        :return:
        """

        self.Dict = json.loads(message)
        #如果判断一段话结束了，就把这段话存储到self.total_sentance里面，下一段话就可以在这一段话之后继续拼接
        if self.Dict['type']=="final_result"  and self.asyn:
            self.total_sentance += json.loads(self.Dict['nbest'])[0]['sentence']
        #server_ready是我们发送开始帧的时候传回来的数据，我们不需要读取
        if self.Dict['type'] != "server_ready" and self.asyn:
            print('\r',self.total_sentance+json.loads(self.Dict['nbest'])[0]['sentence'],end='', flush=True)

    def on_open(self,ws, record_time_s):
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
            if self.asyn: #一边说一边识别
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
                total_time_s = time.time()+total/32000
                while index < total:
                    end = index + chunk_len
                    if end >= total:
                        # 最后一个音频数据帧
                        end = total
                    body = pcm[index:end]
                    ws.send(body, websocket.ABNF.OPCODE_BINARY)
                    index = end
                    time.sleep(chunk_ms / 1000.0)  # ws.send 也有点耗时，这里没有计算
                    last_time = round(total_time_s-time.time(),1)
                    print('\r',"识别中，预计还差",last_time if last_time>0 else 0.00,"秒",end='',flush=True)
            print("\n识别结束")
            #避免时间过短导致句子还没结束，函数就结束了
            if self.Dict:
                if self.Dict["type"] == "partial_result":
                    self.total_sentance += json.loads(self.Dict['nbest'])[0]['sentence']
            # 发送结束帧，写死的，不要动
            endData = '{ signal: "end" }'
            ws.send(endData, websocket.ABNF.OPCODE_TEXT)
            self.ws_app.close()

        threading.Thread(target=run).start()









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


