from control.util import flappy_bird
from multiprocessing import Process
import os
import time

def generateCloud():
    """
    因为词云的import太长了，所以就另外搞一个函数来导入它
    :return:
    """
    from control.util.word_cloud import generateCloud as genC
    genC()


def case_line_patrol():
    """
    跟上面的词云函数一个道理,这个是那个奶茶案例还是什么
    :return:
    """
    from control.util.Voice_line_patrol import case_line_patrol as clp
    clp()


def startFlappyBird():
    """
    启动下坠的小鸟
    :return:
    """
    if not os.path.exists("flappy_bird_temp.cd"):
        with open("flappy_bird_temp.cd", 'w') as file:
            file.write(str(int(time.time())))
        os.environ['SDL_VIDEO_WINDOW_POS'] = "350,50"
        p = Process(target=flappy_bird.man)  # 设置子进程执行的函数，实例化一个对象绑定名称。
        p.start()  # 启动子进程
        os.environ['SDL_VIDEO_WINDOW_POS'] = "50,50"
        flappy_bird.robot()
    else:
        with open("flappy_bird_temp.cd", 'r') as file:
            try:
                last_time = int(file.read())
            except ValueError:
                file.close()
                os.remove("flappy_bird_temp.cd")
                last_time = time.time()
        if int(time.time()) - last_time < 2:
            return
        with open("flappy_bird_temp.cd", 'w') as file:
            file.write(str(int(time.time())))
        os.environ['SDL_VIDEO_WINDOW_POS'] = "350,50"
        p = Process(target=flappy_bird.man)  # 设置子进程执行的函数，实例化一个对象绑定名称。
        p.start()  # 启动子进程
        os.environ['SDL_VIDEO_WINDOW_POS'] = "50,50"
        flappy_bird.robot()


def poetry():
    from control.util.all_path import system_platform
    if 'win' not in system_platform:
        print("树莓派不支持该案例")
        exit()
    print("正在初始化，预计要15秒......")
    from control.util.auto_poetry import poetry_gen, isAllChinese, poetry_show
    # 随机生成一首诗

    print("初始化结束")
    while True:
        head = input("藏头诗，请输入每一行的头，如果留空则随机出诗: ")
        if head == "":
            poetry = poetry_gen.generate()
        elif isAllChinese(head) == False:
            print("输入中含有非中文，请重试")
            continue
        else:
            poetry = poetry_gen.generate(head=list(head))
        poetry_show(poetry)

