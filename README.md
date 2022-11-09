English | [简体中文](README_cn.md)
# control_scnu
![Image text](https://github.com/HuaGuangEdu/control_scnu/blob/main/docs/control1.png)

Developed by HuaGuang AI Education Innovation Team Case Department.

Work for artificial intelligence(AI) education.

# 1.Installation
## Windows & Linux
```python
pip install control-scnu
```
Note that control-scnu is only available for python 3.7 ,3.8, 3.9, 3.10. Python 2.x is not available.

After installing, you should use

```
 import control
```

 instead of 

```
import control-scnu
```

to import the package.

# 2.How to Use: Quick Start

```python
from control import shijue1

a=shijue1.Img()
a.camera(0)
a.name_windows('img')
while True:
    a.get_img()
    a.BGR2GRAY()
    a.GRAY2BIN()
    a.show_image('img')
    a.delay(1)
```

To run the sample, make sure you have a camera in your PC or Raspberry pi.

# 3.File Descriptions

control-scnu contains the file structure as followed:

--control-scnu

​	|--init.py

​	|--file_operation.py

​	|--gpio.py

​	|--jiami.py

​	|--machine_learning.py

​	|--maths.py

​	|--shujue0.py

​	|--shijue1.py

​	|--shijue2.py

​	|--yuyin.py

​	|--unique.py

​	|--lcd

​	|--util

​	|--template


Some examples are as followed. You can read the control-scnu documents for more usages.

See: https://j0tod9knco.feishu.cn/docs/doccn4gtxYau7S2sXzOppDVMOkg

## gpio.py
```
from control import gpio
import time

m=gpio.Mecanum_wheel()
m.uart_init()
m.car_go(200)
time.sleep(2)
m.car_stop()
```

## machine_learning.py
```python
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

## shijue (shijue0,shijue1,shijue2)
```python
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

## yuyin.py
```python
from control import yuyin

s=yuyin.Yuyin(online=True) 
s.my_record(3,"speech")   
print(s.stt("speech"),flush=True)  
s.play_txt(s.stt("speech"))  
