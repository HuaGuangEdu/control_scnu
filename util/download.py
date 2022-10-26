# 作者：tomoya
# 创建：2022-09-30
# 更新：2022-09-30
# 用意：用于从飞书上下载模型之类的大型文件
import py7zr
import os
from control.util import all_path
import requests
import browser_cookie3
import math
from tqdm import trange

models = {
    "win语音识别可执行文件": {
        "fileName": "exeFile",  # 文件夹或者文件名字
        "name": "boxcn1UjUBYR76mRR0IZQUl9BLc",  # 飞书上预览该文件的时候路径的名字
        "size": 3951174,  # 压缩成7z格式之后的文件大小，单位是字节
        "savePath": all_path.speech_path,  # 模型最终保存路径
        "actual_size": 28543677  # 模型解压缩之后的大小
    },
    "linux语音识别可执行文件": {
        "fileName": "exeFile",  # 文件夹或者文件名字
        "name": "boxcnWgvUFyywn2vqh8fTAxD6uc",  # 飞书上预览该文件的时候路径的名字
        "size": 86545,  # 压缩成7z格式之后的文件大小，单位是字节
        "savePath": all_path.speech_path,  # 模型最终保存路径
        "actual_size": 1121900  # 模型解压缩之后的大小
    },
    "k2_rnnt": {
        "fileName": "k2_rnnt",  # 文件夹或者文件名字
        "name": "boxcnRynIDGNTsyYCcVq0gnFjDb",  # 飞书上预览该文件的时候路径的名字
        "size": 307196361,  # 压缩成7z格式之后的文件大小，单位是字节
        "savePath": all_path.speech_path,  # 模型最终保存路径
        "actual_size": 333214810  # 模型解压缩之后的大小
    },
    "conformer": {
        "fileName": "conformer",  # 文件夹或者文件名字
        "name": "boxcni63mB6wpuLzjJyvZ50rXhf",  # 飞书上预览该文件的时候路径的名字
        "size": 427269039,  # 压缩成7z格式之后的文件大小，单位是字节
        "savePath": all_path.speech_path,  # 模型最终保存路径
        "actual_size": 467520352  # 模型解压缩之后的大小
    },
    "stream": {
        "fileName": "stream",  # 文件夹或者文件名字
        "name": "boxcn08bkAsqIGl6eotMMEtbcYg",  # 飞书上预览该文件的时候路径的名字
        "size": 427915699,  # 压缩成7z格式之后的文件大小，单位是字节
        "savePath": all_path.speech_path,  # 模型最终保存路径
        "actual_size": 467526512  # 模型解压缩之后的大小
    },
    "本地化语音": {
        "fileName": "local_yuyin",  # 文件夹或者文件名字
        "name": "boxcnz2bJnn8zUqtSNIw6etkxTb",  # 飞书上预览该文件的时候路径的名字
        "size": 242399826,  # 压缩成7z格式之后的文件大小，单位是字节
        "savePath": all_path.speech_path,
        "actual_size": 417404494
    },
    "自动生成古诗": {
        "fileName": "autoPoetry",
        "name": "boxcnnyV11jZ1yzG1S2aH1uU8vg",
        "size": 468305550,
        "savePath": all_path.model_path,
        "actual_size": 710136756
    },
}
# 用于指定浏览器
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36 Edg/100.0.1185.50',
}


def download(modelName):
    """
    从飞书上下载模型文件
    :param modelName:
    :return:
    """
    # if not modelName in models.keys():
    #     raise NameError("没有这个模型")
    modelDict = models[modelName]
    savePath = modelDict["savePath"]
    modelSize = modelDict["size"]
    downloadPath = f"https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/all/{modelDict['name']}/?mount_point=explorer"
    cj = browser_cookie3.load()
    r = requests.get(downloadPath, cookies=cj, stream=True, headers=headers)
    file = open(os.path.join(savePath, modelDict["fileName"] + ".7z"), 'wb')
    batchsSize = 1000000
    batchsNum = math.ceil(modelSize / batchsSize)
    for i in trange(batchsNum):
        file.write(r.raw.read(1000000))
    file.flush()
    file.close()
    print("解压中")
    with py7zr.SevenZipFile(os.path.join(savePath, modelDict["fileName"] + ".7z"), mode='r') as z:
        z.extractall(savePath)
    print("解压成功")
    os.remove(os.path.join(savePath, modelDict["fileName"] + ".7z"))


def getFileSize(filePath, size=0):
    """
    获取文件夹的总大小
    :param filePath:
    :param size:
    :return:
    """
    if os.path.isfile(filePath):
        size = os.path.getsize(filePath)
    else:
        for root, dirs, files in os.walk(filePath):
            for f in files:
                size += os.path.getsize(os.path.join(root, f))
    return size
