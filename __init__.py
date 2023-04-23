"""
更新内容一览
"""

"""
2022/04/23
没有更新，只是发现有些更新日志没有写上去。现在已经是0.8.6了，希望下一任能够继续完善本库，善待本库，有时间就每周一更。更新完记得github也更上去，别版本错乱了
"""

"""
2023/03/04
control_version:0.8.3
blcoklyFile_version:2.2.11
blockly_version:2.2.6
兼容了中文路径保存图片

"""
"""
2023/01/09
control_version:0.8.2
blcoklyFile_version:2.2.11
blockly_version:2.2.6
1、修改了视觉库shijue0中保存图片的函数，现在支持图片保存成中文名了
"""

"""
2022/10/26
control_version:0.8.1
blcoklyFile_version:2.2.11
blockly_version:2.2.5
1、修改了shijue1中替换背景的那个函数,如果输入的背景图不存在,就默认使用白色背景而不是报错
"""


"""
2022/10/26
control_version:0.8.0
blcoklyFile_version:2.2.11
blockly_version:2.2.5
1、将Yuyin类拆成三个类（语音识别，语音合成，聊天机器人）
2、加入本地化语音识别
3、更新了语音的编程块,以及整理了所有的编程块
4、修改util里面的local_yuyin和all_path和checkLibVersion,download
5、树莓派语音识别测试通过
"""


"""
2022/10/10
control_version:0.7.3
blcoklyFile_version:2.2.10
blockly_version:2.2.5
1、加入树莓派版本的本地化语音合成
2、调整了requirements
3.GPIO库加入了RGB三色灯的应用
4、GPIO三色灯优化逻辑与命名
5、改LCD字体
"""


"""
2022/10/07
control_version:0.7.2
blcoklyFile_version:2.2.10
blockly_version:2.2.5
1、整理了control库，增加了util文件夹存放各种工具
2、增加了案例体验这个分支，增加了本地化语音、启动古诗生成器编程块，将词云编程块从画图转移到文本，将语音控制小车编程块从硬件控制转移到案例体验
3、删除了integration.py并在readme中同步删除了跟integration.py有关的信息
4、删除了lcd文件夹中没有用到的py文件
5、更新了requirements
"""

"""
2022/9/22
control_version:0.7.1
blcoklyFile_version:2.2.9
blockly_version:2.2.2
1、修改gpio中led，修正亮灯为低电平（硬件更换了）
"""

"""
2022/9/16
control_version:0.7.0
blcoklyFile_version:2.2.9
blockly_version:2.2.2
1、机器学习库重构（不过保留了之前的，以兼容以前版本的编程块）
    影响到的库有machine_learning和unique
2、blocklyFiles中的机器学习编程块做出部分改动
3、增加颜色聚类的代码和块
"""


"""
2022/9/3
control_version:0.6.9
blcoklyFile_version:2.2.8
blockly_version:2.2.2
修复了已知bug

2022/8/27
control_version:0.6.8
blcoklyFile_version:2.2.8
blockly_version:2.2.2
增加了语音的本地化文字转语音，并修改了语音的块

2022/7/15
control_version:0.6.7
blcoklyFile_version:2.2.7
blockly_version:2.2.2
blockly_version:2.2.2
control库增加了一个maths模块

blocklyFiles
1、重写了数学、逻辑、控制三个分支内的部分编程块
2、删除了语音的js文件内重复出现的定义函数
3、删除了输入/输出分支的js文件内无用的定义函数

"""

"""
2022/7/3
control_version:0.6.7
blcoklyFile_version:2.2.7
blockly_version:2.2.2
shijue1增加了两个函数
1、创建一个与输入图像相同尺寸的全零图像
2、输入图像和两点，在图像的这两点之间画线

unique增加了一个函数和一个类，用于词云案例

blocklyFiles
1、在视觉的基本操作那里加入了四个块，用于蓝线挑战案例
2、在画图里面加入了一个块，用于词云案例
"""

"""
2022/7/2
control_version:0.6.6
blcoklyFile_version:2.2.7
blockly_version:2.2.2
增加深度学习的分支。
增加“当前图像”这个块。

"""

"""
2022/7/2
control_version:0.6.5
blcoklyFile_version:2.2.6
blockly_version:2.2.2
修复图像resize错对象问题，原有会不显示处理后图像。
修复paddlelite问题。
修复依赖bug。

"""

"""
2022/6/2
control_version:0.6.4
blcoklyFile_version:2.2.6
blockly_version:2.2.2
路径问题（最终）

"""


"""
2022/5/30
control_version:0.6.3
blcoklyFile_version:2.2.6
blockly_version:2.2.2
基本稳定了本地化语音

"""

"""
2022/5/29
control_version:0.6.2
blcoklyFile_version:2.2.6
blockly_version:2.2.2
修改路径bug

"""

"""
2022/5/23
control_version:0.6.1
blcoklyFile_version:2.2.6
blockly_version:2.2.2
修改一个视觉bug

"""

"""
2022/5/23
control_version:0.6.0
blcoklyFile_version:2.2.6
blockly_version:2.2.2
本次修改更新较多，主要内容如下：
1.增加了本地化语音
2.增加了深度学习模块
"""

"""
2022/5/23
control_version:0.5.12
blcoklyFile_version:2.2.6
blockly_version:2.2.2
1、禁用了树莓派上使用本地化语音转文字
2、优化了语音库的路径
"""

"""
2022/5/22
control_version:0.5.12
blcoklyFile_version:2.2.6
blockly_version:2.2.2
1、解决了树莓派上不能用语音的问题，树莓派上将使用mplayer来播放音频，win上使用playsound
2、语音转文字中，删掉了文字中的中文数字转成阿拉伯数字时的print(numStr)
"""
        

"""
2022/5/20
control_version:0.5.12
blcoklyFile_version:2.2.6
blockly_version:2.2.2
1、解决了路径问题。
2、部署了本地化语音到语音库。
3、显示窗口放大回640 x 320分辨率。
"""

"""
2022/5/18
control_version:0.5.12
blcoklyFile_version:2.2.6
blockly_version:2.2.2
改了所有库的注释以及参数的类型注解
"""


"""
2022/5/11
control_version:0.5.12
blcoklyFile_version:2.2.6
blockly_version:2.2.2
1.GPIO超声波返回值为两位小数的值，单位为cm
2.改了语音的路径
"""

"""
2022/5/11
control_version:0.5.11
blcoklyFile_version:2.2.6
blockly_version:2.2.2
忘了修改了什么

"""

"""
2022/5/10
control_version:0.5.10
blcoklyFile_version:2.2.6
blockly_version:2.2.2
修改视觉和机器学习
"""

"""
2022/5/9
control_version:0.5.9
blcoklyFile_version:2.2.6
blockly_version:2.2.2
修改舵机
"""

"""
2022/5/5
control_version:0.5.8
blcoklyFile_version:2.2.6
blockly_version:2.2.2
更新内容
1、增加了兼容绝对路径和相对路径（适用于所有库）
2、yuyin库大幅度修改
3、unique库移植进了修改后的playsound库
"""


"""
2022/4/30
control_version:0.5.7
blcoklyFile_version:2.2.5
blockly_version:2.2.2
更新内容
1.加入摄像头判定语句，若为Win系统则不进行缩放（电脑算力足够）
2.修改显示图像resize逻辑，直接写为640，320大小。（之前是放大两倍存在逻辑问题）
更改了了blockly读取图片逻辑奇怪的问题
control对应加入了一个self.picture参数判断
"""

"""
2022/4/29
control_version:0.5.6
blcoklyFile_version:0.2.4
blockly_version:2.3.0
更新内容：
修复了语音的若干个bug
"""

"""
2022/4/20
control_version:0.5.5
blcoklyFile_version:0.2.4
blockly_version:2.2.2
更新内容：
修复gpio无法直接使车辆停止的bug
修复m.stop（）必须要加一个小前进速度的bug
删除了没有的改变声音的语言块
"""

"""
2020/4/4
control_version:0.5.4
blcoklyFile_version:0.2.3
blockly_version:2.2.1
更新内容：
修改了视觉部分，将摄像头分辨率降低到320✖240。但是最后的输出还是放大回了640✖480
修改了舵机的块

"""

"""
2022/4/1
control_version:0.5.3
blcoklyFile_version:0.2.3
blockly_version:2.2.1
更新内容：
修改了machine_learning的大量内容
修改了机器学习对应的块
增加了模版匹配函数
增加了模版匹配的块
"""

"""
2022/3/27
control_version:0.5.2
blcoklyFile_version:0.2.2
blockly_version:2.2.1
更新内容：
修改了blocklyFiles的块：硬件控制-->小车控制-->中的第二个块
增加了屏幕函数
修改了仿真器的块
"""

"""
2022/3/21
control_version:0.5.1
blcoklyFile_version:0.2.0
blockly_version:2.2.1
更新内容：
修改了在树莓派不能调用摄像头的bug
"""

"""
2022/3/21
control_version:0.5.0
blcoklyFile_version:0.2.0
blockly_version:2.2.1
更新内容：
1.修改了人脸识别与数字识别的块和库
2.修改了镜像翻转的操作
3.修改了视觉的块，现在视觉的类分成了三个小类
4.增加了颜色形状的检测和识别
5.修改了mediapipe的库和类，包括身体部位检测、手指检测
6.删除了原来有的人脸检测，换成了mediapipe的
7.增加了背景图像切换的操作
"""

"""
2022/3/13
control_version:0.4.2
blcoklyFile_version:0.1.2
blockly_version:2.2.1
更新内容：
1.修改了数字识别模型
2.修改了数字对应的块
3、shijue0库加入了两个函数，查找轮廓和返回图像类型。
4、shijue0删除了一个变量self.Type，原本是用来返回图像类型的，现在被上面那个新函数代替。
5、增加了轮廓检测的块
6、修改了返回图像类型的块的代码区域的内容
"""

"""
2022/3/7
control_version:0.4.1
blcoklyFile_version:0.1.1
blockly_version:2.2.1
"""
