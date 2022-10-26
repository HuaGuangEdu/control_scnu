# 作者：tomoya
# 创建：2019-10-21
# 更新：2019-10-21
# 用意：用于检查control库是否为最新
import os
import threading
import time
import requests


class LibVersionChecker(threading.Thread):
    def __init__(self):
        super().__init__()
        pass

    def getLatestLib(self):
        """
        获取清华源上最新的control库版本
        :return:
        """
        res = requests.get("https://pypi.tuna.tsinghua.edu.cn/simple/control-scnu/")
        result = str(res.content)
        begin = result.rfind("control_scnu-")
        end = result.rfind(".tar.gz")
        return result[begin:end].replace("control_scnu-", "")

    def getLibVersion(self):
        """
        获取当前control库版本
        :return:
        """
        result = os.popen("pip show --files control-scnu")
        result = result.buffer.read().decode('utf-8')
        begin = result.find("Version:")
        end = result.find("Summary", begin)
        return result[begin:end].replace("Version:", "").replace(" ", "").replace("\r\n", "")

    def run(self):
        """
        如果当前库版本和最新的版本不一致，就提示用户更新control库，这个提醒每个一个小时提醒一次
        加个try except语句是为了防止子线程报错导致主线程也跟着报错
        :return:
        """
        try:
            if os.path.exists(".LibVersionChecker"):
                with open(".LibVersionChecker", 'r') as f:
                    if f.read() == "".join(map(str, time.localtime()[0:4])):
                        return
            nowLibVersion = self.getLibVersion()
            latestLibVersion = self.getLatestLib()
            result = nowLibVersion != latestLibVersion
            if result:
                print(f"warning: 当前control库的版本为{nowLibVersion}不是最新的，你可以运行 pip install control-scnu -U 来更新至{latestLibVersion}")
            with open(".LibVersionChecker", 'w') as f:
                f.write("".join(map(str, time.localtime()[0:4])))
        except:
            pass



if __name__ == '__main__':
    t1 = LibVersionChecker()
    t1.run()
