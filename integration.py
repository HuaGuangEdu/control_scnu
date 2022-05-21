from control import yuyin
from control import gpio
from control import shijue1
import time
from typing import Any

def case_line_patrol(m:Any):  #单圈单白线语音巡线高集成度函数
    s=yuyin.Yuyin()
    condition = 0  #判断是否进入巡线，防止一次语音没回答到


    while True:
        if condition == 0:
            m.xunxian(0,1)  #设置左右红外io口
            condition = 1
            #下面是让小车走直线一点点距离，越过白线再次回到起点（仅仅针对单圈）
            m.car_contr(150, 0, 0)
            time.sleep(2)
            m.car_contr(0,0,0)
        s.tts("寻线完毕请问接下来要做什么","audio")
        s.play_music("audio")
        s.my_record(3,"speech")
        print(s.stt("speech"))
        if "继" in s.stt("speech"):
            condition = 0
        if "结" in s.stt("speech"):
            break
    s.tts("程序结束","audio")
    s.play_music("audio")
