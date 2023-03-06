import cv2
import numpy as np
from .util.all_path import picture_path,model_path,system_platform

is_windows = 1 if 'win' in system_platform else 0

if not is_windows:
    try:
        import paddlelite.lite
    except:
        print('没有安装paddlelite。请到官网下载树莓派镜像源，并按说明书操作')

def check_model(model):
    if model[-2:] != 'nb' and model[-4:] != 'onnx':
        raise TypeError('模型格式不对，确保你的模型格式是.nb或.onnx格式的')
    elif model[-2:] != 'nb' and is_windows == 0:
        raise TypeError('模型格式不对，在树莓派或其他设备上只能用.nb格式的')
    elif model[-4:] != 'onnx' and is_windows == 1:
        raise TypeError('模型格式不对，在windows上只能用.onnx格式的')


class AdvancedImg:
    def __init__(self):
        self.img = None
        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

    def process_image(self, image_data, shape=64, standard=True, black=False):
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        if black:
            gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            ret, image_data = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
            cv2.imshow('g',image_data)
            cv2.waitKey(1)
            image_data = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
        image_data = cv2.resize(image_data, (shape, shape))
        image_data = image_data.transpose((2, 0, 1)) / 255.0
        if standard:
            image_data = (image_data - np.array(self.img_mean).reshape(
                (3, 1, 1))) / np.array(self.img_std).reshape((3, 1, 1))
        image_data = image_data.reshape([1, 3, shape,shape]).astype('float32')
        return image_data

    def classify_number_init(self):
        if is_windows:
            model = 'numbers.onnx'
            paddle_model = model_path + model
            self.predictor = cv2.dnn.readNetFromONNX(paddle_model)
        else:
            model = 'numbers.nb'
            paddle_model = model_path + model
            config = paddlelite.lite.MobileConfig()
            config.set_model_from_file(paddle_model)
            self.predictor = paddlelite.lite.create_paddle_predictor(config)
            self.input_tensor0 = self.predictor.get_input(0)

    def detect_pingpong_init(self):
        if is_windows:
            raise ImportError('windows下暂不支持目标检测，请充值VIP后再试')
            model = 'pingpong.onnx'
            paddle_model = model_path + model
            self.predictor = cv2.dnn.readNetFromONNX(paddle_model)
        else:
            model = 'pingpong.nb'
            paddle_model = model_path + model
            config = paddlelite.lite.MobileConfig()
            config.set_model_from_file(paddle_model)
            self.predictor = paddlelite.lite.create_paddle_predictor(config)
            self.input_tensor0 = self.predictor.get_input(0)
            self.input_tensor1 = self.predictor.get_input(1)

    def classify_model_init(self, model='numbers.nb'):
        check_model(model)
        paddle_model = model_path + model
        if not is_windows:
            config = paddlelite.lite.MobileConfig()
            config.set_model_from_file(paddle_model)
            self.predictor = paddlelite.lite.create_paddle_predictor(config)
            self.input_tensor0 = self.predictor.get_input(0)
        else:
            self.predictor = cv2.dnn.readNetFromONNX(paddle_model)

    def detect_model_init(self, model='pingpong.nb'):
        check_model(model)
        paddle_model = model_path + model
        if not is_windows:
            config = paddlelite.lite.MobileConfig()
            config.set_model_from_file(paddle_model)
            self.predictor = paddlelite.lite.create_paddle_predictor(config)
            self.input_tensor0 = self.predictor.get_input(0)
            self.input_tensor1 = self.predictor.get_input(1)
        else:
            self.predictor = cv2.dnn.readNetFromONNX(paddle_model)

    def infer_number(self):
        input_c, input_h, input_w = 3, 128, 128
        temp = cv2.imread(picture_path + 'temp_number.jpg')
        temp = cv2.resize(temp, (130,130))
        res = cv2.matchTemplate(self.img, temp, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        w, h = temp.shape[1], temp.shape[0]
        top_left = max_loc
        if max_val > 0.28:
            cut_img = self.img[top_left[1]:top_left[1] + h, top_left[0]: top_left[0] + w]
            image_data = self.process_image(cut_img, standard=False, black=True)
            if not is_windows:
                self.input_tensor0.from_numpy(image_data)
                self.predictor.run()
                output_tensor = self.predictor.get_output(0)
                output_tensor = output_tensor.numpy()
            else:
                self.predictor.setInput(image_data)
                output_tensor = self.predictor.forward()

            e_x = np.exp(output_tensor.squeeze() - np.max(output_tensor.squeeze()))
            pro = e_x / e_x.sum()
            if np.max(pro) > 0.5:
                classnum = np.argmax(pro)
                self.m_data = classnum
            else:
                self.m_data = -1
        else:
            self.m_data = -1

    def infer_pingpong(self):
        label, x1, y1, x2, y2 = 'None',0,0,0,0
        datalist = []
        input_h, input_w, input_c = self.img.shape
        img2 = self.img.copy()
        image_data = self.process_image(img2, shape=128)
        if not is_windows:
            put1 = np.array([128/input_h, 128/input_w])
            put1 = put1.reshape([1, 2]).astype('float32')
            self.input_tensor0.from_numpy(image_data)
            self.input_tensor1.from_numpy(put1)
            self.predictor.run()
            output_tensor = self.predictor.get_output(0)
        else:
            self.predictor.setInput(image_data)
            output_tensor = self.predictor.forward()

        output_data = output_tensor.numpy()
        for i in output_data:
            if i[1] >= 0.3:
                label, pro, x1, y1, x2, y2 = i
                label = 'ball'
                datalist.append([label,(x1,y1),(x2,y2)])
                cv2.rectangle(self.img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        self.m_data = label
        self.datalist = datalist

    def infer_classify(self):
        raise ImportError('该功能暂未开放，请充值VIP后再试')

    def infer_detect(self):
        raise ImportError('该功能暂未开放，请充值VIP后再试')
