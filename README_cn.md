
- [0.control_scnu](#0control_scnu)
- [1.安装control库](#1安装control库)
- [2.各个对应文件描述](#2各个对应文件描述)
  - [template](#template)
  - [_init_.py](#initpy)
  - [creat_img.py](#creat_imgpy)
  - [file_operation.py](#file_operationpy)
  - [gpio.py](#gpiopy)
  - [integration.py](#integrationpy)
  - [jiami.py](#jiamipy)
  - [machine_learning.py](#machine_learningpy)
  - [maths.py](#mathspy)
  - [requirements.txt](#requirementstxt)
  - [shijue (shijue0,shijue1,shijue2)](#shijue-shijue0shijue1shijue2)
  - [unique](#unique)
  - [yuyin.py](#yuyinpy)
# 0.control_scnu
该库属于 华光人工智能教育创新团队

# 1.安装control库
```python
pip3 install control-scnu
```

# 2.各个对应文件描述
## template
各种模型文件
## _init_.py
init文件，里面使用注释写明了软件、编程快、库的版本信息
## creat_img.py

## file_operation.py
文件操作相关库

## gpio.py
小车硬件相关库

## integration.py
用于高集成化案例的库

## jiami.py
加密文件的库

## machine_learning.py
机器学习的库

## maths.py
基本数学相关的库

## requirements.txt
库的依赖txt文件

## shijue (shijue0,shijue1,shijue2)

## unique

## yuyin.py
语音相关库
语音识别并复述的例程
```
from control import yuyin

s=yuyin.Yuyin(online=True)  #实例化语音，选择在线识别模式
s.my_record(3,"speech")   #录音3秒保存到 speech
print(s.stt("speech"))  #打印识别的文字
s.play_txt(s.stt("speech"))  #复述识别的文字

image.png