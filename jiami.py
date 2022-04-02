"""
1、安装Cython，速度过慢可以换源来安装
$sudo pip3 install Cython
$sudo pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple cython

2、加密需要对应环境系统下加密
Cython加密python库生成so或pyd文件，在linux下生成so，在win下生成pyd

3、文件路径
file_name为打算加密的py文件
build_dir是生成的加密so文件的文件夹
build_tmp_dir是临时文件夹，会生成o文件

4、一些额外注意的东西
py文件同目录下会有py转化的c文件，c文件/o文件可以删除，同时pyc文件也可删除
so文件中间的词缀可以删掉，只保留py同名也可使用
当同名so和py在同一个文件夹时，代码会优先调用so文件
"""

# coding:utf-8
from distutils.core import setup
from Cython.Build import cythonize
import sys

file_name = ['gpio.py',
             'shijue1.py',
             'file_operation.py',
             'create_img.py',
             'machine_learning.py',
             'yuyin.py',
             'shijue0.py',
             'integration.py',
             'unique.py']


if __name__ == "__main__":
    system_platform = sys.platform
    build_dir = "/home/pi/Desktop/mypack"
    build_tmp_dir = "/home/pi/Desktop/mypack"
    if 'win' in system_platform:
        build_dir = "./mypack"
        build_tmp_dir = "./mypack"
    for f in file_name:
        setup(ext_modules=cythonize(f),script_args=["build_ext","-b",build_dir,"-t",build_tmp_dir])
