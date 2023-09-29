# 作者：tomoya
# 创建：2022-09-30
# 更新：2022-09-30
# 用意：用于存放各个文件夹地址
import os
import sys
import warnings
from control.util.checkLibVersion import LibVersionChecker
from control import util
# 查看当前control库是否为最新版本
LibVersionChecker().start()
warnings.filterwarnings("ignore")
system_platform = sys.platform

if 'win' in system_platform:
    # 获取当前文件的位置，就是有我们软件的exe的那个路径
    file_path = os.path.join(os.getcwd().split('blockly-electron')[0], 'blockly-electron')
    if not os.path.exists(file_path):
        # 有时候拿到的是免安装版本的，就会出现没有blockly-electron这个文件夹
        if os.path.exists(os.path.join(os.getcwd(), "resources")):
            file_path = os.getcwd()
    class_path = os.path.join(file_path, 'resources', 'assets', 'class').replace("\\", "/")
else:
    class_path = '/home/pi/class/'  # 树莓派的class文件夹地址
data_path = os.path.join(class_path, 'data')
decorate_path = os.path.join(class_path, 'decorate')
emulator_files_path = os.path.join(class_path, 'emulator_files')
file_operation_path = os.path.join(class_path, 'file_operation')
fonts_path = os.path.join(class_path, 'fonts')
model_path = os.path.join(class_path, 'model')
picture_path = os.path.join(class_path, 'picture')
speech_path = os.path.join(class_path, 'speech')
txt_path = os.path.join(class_path, 'txt')

# 新的cheakpath
def checkPathExists(path):
    if not os.path.exists(path):  # 如果已经有了是不会创建目录的
        os.makedirs(path, exist_ok=True)

# 检查并创建目录
checkPathExists(class_path)



for path in [class_path, data_path, decorate_path, emulator_files_path, file_operation_path, fonts_path, model_path,
             picture_path,
             speech_path, txt_path]:
    checkPathExists(path)

digit_template_dir = util.__file__.replace("__init__.py","something\\template.npy")
