from PIL import Image,ImageDraw,ImageFont
import os
import random

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f',
            'g', 'h', 'i', 'j', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's',
            't', 'u', 'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
            'H', 'I', 'J', 'L', 'M', 'N', 'O',
            'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y', 'Z']
font_type = ['1', '2', '3', '4', '5']
angle_type = [0]
#angle_type=[0,90,180,270]
image_size = 64
fontsize = 50


# 随机选取颜色
def getRandomColor():
    '''
    随机选取颜色
    Returns:

    '''
    c1 = random.randint(0, 255)
    c2 = random.randint(0, 255)
    c3 = random.randint(0, 255)
    return (c1, c2, c3)


# 随机选取字体
def getfont():
    '''
    随机获取字体
    Returns:

    '''
    font = random.sample(font_type, 1)
    fontpath='../class/fonts/'+font[0]+'.ttf'
    # print(fontpath)
    return fontpath


# 随机选取旋转角度
def getangle():
    '''
    随机获取旋转角度
    Returns:

    '''
    angle = random.sample(angle_type, 1)
    # print(angle[0])
    return angle[0]


# 随机位置
def getlocation():
    '''
    获取随机位置
    Returns:

    '''
    location = (random.randint(0, image_size-fontsize),
                random.randint(0, image_size-fontsize))
    # print(location)
    return (location)


def creat_number_img(num:int, path:str):
    '''
    生成数字图像
    Args:
        num: 生成的数字
        path: 生成的位置

    Returns:

    '''
    for j in range(num):
        for i in number:
            path_ = path+'/'+i
            if not os.path.exists(path_):
                os.mkdir(path_)
            image = Image.new(mode='RGB', size=(image_size, image_size), color=getRandomColor())
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(getfont(), size=fontsize)
            random_char = i
            draw.text(getlocation(), random_char, getRandomColor(), font=font)
            image.rotate(getangle()).save(path_+'/ {}.jpg'.format(j))
    print('图片生成完毕')

#生成数字图像
# creat_number_img(10,r'..\class\data\num_pic')
