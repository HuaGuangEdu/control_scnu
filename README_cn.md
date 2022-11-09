
[English](README.md) | 简体中文
- [0.control_scnu](#0control_scnu)
- [1.安装control库](#1安装control库)
- [2.各个对应文件描述](#2各个对应文件描述)
  - [template](#template)
  - [_init_.py](#initpy)
  - [file_operation.py](#file_operationpy)
  - [gpio.py](#gpiopy)
  - [jiami.py](#jiamipy)
  - [machine_learning.py](#machine_learningpy)
  - [maths.py](#mathspy)
  - [requirements.txt](#requirementstxt)
  - [shijue (shijue0,shijue1,shijue2)](#shijue-shijue0shijue1shijue2)
  - [unique.py](#uniquepy)
  - [yuyin.py](#yuyinpy)
# 0.control_scnu
该库为 华光人工智能教育创新团队 [案例部] 开发  
适用于人工智能教育

# 1.安装control库
## windows
```python
pip install control-scnu
```
## linux
```python
pip3 install control-scnu
```

# 2.各个对应文件描
## _init_.py
init文件  
里面使用注释写明了软件、编程块、库的版本信息

## file_operation.py
文件操作相关库

## gpio.py
小车硬件相关库  
小车前进的例程（该项需要在小车上运行）:
```
from control import gpio
import time

m=gpio.Mecanum_wheel()
m.uart_init()
m.car_go(200)
time.sleep(2)
m.car_stop()
```


## jiami.py
加密文件的库

## machine_learning.py
机器学习的库  
鸢尾花机器学习例程:
```
from control import machine_learning as ml

datasets=ml.DatasetsNew(ml.data_name["鸢尾花"])
model= ml.ModelNew(ml.model_name['神经网络'])
model.train(datasets.x_train, datasets.y_train,dataName=datasets.data_name)
model.test(datasets.x_test,datasets.y_test)
print(model.test_score,flush=True)
model.predict(datasets.x_test)
print(model.pred,flush=True)
model.save(name='myFirstModel')
model1=ml.ModelNew('myFirstModel.proto')
model1.test(datasets.x_test,datasets.y_test)
print(model.pred,flush=True)
```

## maths.py
基本数学相关的库

## requirements.txt
库的依赖txt文件

## shijue (shijue0,shijue1,shijue2)
视觉相关的库  
摄像头获取图像并二值化显示例程:
```
from control import shijue1

a=shijue1.Img()
a.camera(0)
a.name_windows('img')
while  True:
    a.get_img()
    a.BGR2GRAY()
    a.GRAY2BIN()
    a.show_image('img')
    a.delay(1)
```

## unique.py
放一些特别的东西

## yuyin.py
语音相关库  
语音识别并复述的例程:
```
from control import yuyin

s=yuyin.Yuyin(online=True) 
s.my_record(3,"speech")   
print(s.stt("speech"),flush=True)  
s.play_txt(s.stt("speech"))  
