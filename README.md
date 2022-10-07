English | [简体中文](README_cn.md)
- [0.control_scnu](#0control_scnu)
- [1.Install control](#1install-control)
- [2.Each corresponding file description](#2each-corresponding-file-description)
  - [template](#template)
  - [_init_.py](#initpy)
  - [file_operation.py](#file_operationpy)
  - [gpio.py](#gpiopy)
  - [integration.py](#integrationpy)
  - [jiami.py](#jiamipy)
  - [machine_learning.py](#machine_learningpy)
  - [maths.py](#mathspy)
  - [requirements.txt](#requirementstxt)
  - [shijue (shijue0,shijue1,shijue2)](#shijue-shijue0shijue1shijue2)
  - [unique.py](#uniquepy)
  - [yuyin.py](#yuyinpy)
# 0.control_scnu
The library was developed for Huaguang AI Education Innovation Team [Case Department]    
Applicable to artificial intelligence education

# 1.Install control
```python
pip3 install control-scnu
```

# 2.Each corresponding file description
## template
Where the various model files are saved

## _init_.py
The init file  
The version information of the software, programming block and library is indicated in this file

## file_operation.py
File manipulation related libraries

## gpio.py
Car hardware related library 
The car forward routine (This item needs to run on the car):
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
A library for encrypted files

## machine_learning.py
Library for machine learning    
Iris machine learning routine:
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
Library related to basic mathematics

## requirements.txt
Library dependent TXT file

## shijue (shijue0,shijue1,shijue2)
Visual related libraries   
The camera obtains the image and binarizes the display routine:
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
Put something special in it

## yuyin.py
Speech correlation library    
Routines for speech recognition and retelling:
```
from control import yuyin

s=yuyin.Yuyin(online=True) 
s.my_record(3,"speech")   
print(s.stt("speech"),flush=True)  
s.play_txt(s.stt("speech"))  
