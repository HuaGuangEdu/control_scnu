# 作者：tomoya
# 创建：2022-09-29
# 更新：2022-09-29
# 用意：本地化语音转文字专用的类
import os
import subprocess
import websocket
import pyaudio
import time
import json
import threading
from control.util.all_path import speech_path, system_platform


class Yuyin_local():
    def __init__(self, record_time_s, local_yuyinPath, asyn=True, filename=''):
        os.system("chcp 65001")  # 切换语音为Unicode (UTF-8)
        self.local_yuyinPath = local_yuyinPath
        # vbsPath = os.path.join(local_yuyinPath, "runbat.vbs")
        # subprocess.call(f"cscript  {vbsPath}", stdout=None, stdin=None)
        self.record_time_s = record_time_s
        self.ws_app = websocket.WebSocketApp("ws://127.0.0.1:10086",
                                             on_open=lambda ws: self.on_open(ws, record_time_s),  # 连接建立后的回调
                                             on_message=self.on_message,  # 接收消息的回调
                                             on_error=self.on_error,
                                             on_close=self.on_close,
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
        self.isError = False

    def run(self):
        self.ws_app.run_forever()
        return self.isError

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
            """
             发送二进制音频数据，注意每个帧之间需要有间隔时间
             :param ws:
             :param record_time_s: 录音时长，单位是秒
             :return:
             """
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

    def on_error(self, ws, error):
        if "由于目标计算机积极拒绝，无法连接" in str(error) or "强迫关闭" in str(error):
            vbsPath = os.path.join(self.local_yuyinPath, "runbat.vbs")
            subprocess.call(f"cscript  {vbsPath}", stdout=None, stdin=None)
            self.isError = True
        else:
            print(error)

    def on_close(self, we):
        pass

    def on_data(self, we, message, message_len, isSend):
        pass

    def on_ping(self):
        pass

    def on_pong(self):
        pass

    def on_cont_message(self):
        pass


class Yuyin_local2():
    def __init__(self, modelName, wavFile):
        self.modelName = modelName
        # win的可执行文件有个.exe后缀，linux的可执行文件没有
        self.exe_path = os.path.join(speech_path, "exeFile", modelName + (".exe" if "win" in system_platform else ""))
        self.model_path = os.path.join(speech_path, modelName)
        self.wav_path = os.path.join(speech_path, wavFile)

    def recognize(self):
        # 调用可执行文件
        if "win" not in system_platform:
            # 在linux上可能需要给这个可执行文件添加可执行权限，不然后面会报权限不足
            os.system("sudo chmod +x "+self.exe_path)
        result = os.popen(" ".join([self.exe_path, self.model_path, self.wav_path]))
        if self.modelName == 'stream':
            # 流式语音识别
            while True:
                nowResult = result.buffer.readline().decode("utf8")
                if nowResult == '':
                    break
                elif "当前识别结果" in nowResult:
                    print('\r', nowResult.replace("\r\n", "").replace("当前识别结果:  ", "").replace('"', "")[:-1], end='',
                          flush=True)
                elif "最终结果" in nowResult:
                    final = nowResult
            print("\n")
            return final.replace('最终结果为: "', "")[:-4]
        # 非流式语音识别
        result = result.buffer.read().decode('utf-8')  # 将输出以utf8解码，这样就不会报错
        result = result[result.find("识别结果") + 7:result.rfind("识别时间") - 1]  # 从输出中选择语音识别的结果，不需要其他信息
        return result
