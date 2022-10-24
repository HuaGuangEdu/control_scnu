# 作者：tomoya
# 创建：2022-09-29
# 更新：2022-09-29
# 用意：数字转换，将中文数字转换成阿拉伯数字，用于语音识别时将识别到的数字转换成阿拉伯数字
import re


class Number_Convert():
    def __init__(self):

        self.number_map = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8,
                           '九': 9}  # 1-9数字
        self.unit_map = {'十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}  # 数字单位

    def __operate(self, num_str: str):  # 这个和下面呢个____operate1都是处理字符串的函数，别调用
        Num = 0
        unit = False
        for index, i in enumerate(num_str[::-1]):
            if i in self.number_map.keys():
                Num += self.number_map[i] * (unit if unit else 1)
            elif index != len(num_str) - 1:
                unit = self.unit_map[i]
            else:
                Num += self.unit_map[i]
        return Num

    def __operate1(self, strings: str):  # 处理字符串的，分成了三种情况，有“亿”，无“亿”有“万”， 无“亿”无“万”

        if '亿' in strings:
            strings2 = strings.split('亿')
            Num1 = 0
            for index0, i in enumerate(strings2):
                Num = 0
                if len(i.split('万')) != 1:
                    for index, j in enumerate(i.split('万')):
                        Num += self.__operate(j) * (10000 if index == 0 else 1)
                else:
                    Num += self.__operate(i.split('万')[0])

                Num1 += Num * (self.unit_map['亿'] if index0 == 0 else 1)
            return Num1
        elif '万' in strings:
            Num = 0
            for index, j in enumerate(strings.split('万')):
                Num += self.__operate(j) * (self.unit_map['万'] if index == 0 else 1)
            return Num
        else:
            return self.__operate(strings)

    def num_convert3(self, test_strings: str):
        self.NumList = []  # 装数字的列表
        self.converted_strings = ''  # 转化后的字符串
        self.test_strings = test_strings.replace("什", "【·&……】")
        for index0, Str in enumerate(self.test_strings):  # 遍历一下字符串
            try:  # 如果已经遍历完一串数字，那我们要把当前的位置移动到这一串数字之后，再继续遍历下面的内容
                if index0 < index1:
                    continue
            except:  # 如果报错了，表示还没有遍历过任何一串数字，所以j还没有定义
                pass
            if (Str.isnumeric() and not Str.isdigit()) or (
                    Str == '两'):  # 如果遍历到的那个字符是中文数字，不是阿拉伯数字，那就从那个字符开始，遍历那个字符以及之后的字符串部分
                for index1, Str2 in enumerate(self.test_strings[index0:]):
                    if (not Str2.isnumeric()) and Str2 != '两':  # 如果遍历到不是数字的字符，表示这一串数字遍历完了，那就开始将这一串中文数字转化成阿拉伯数字
                        Num = self.__operate1(self.test_strings[index0:index0 + index1])
                        self.converted_strings += str(Num)
                        self.NumList.append((Num, index0))
                        index1 = index0 + index1  # 让index1表示当前遍历到的字符串的位置
                        break
            else:
                self.converted_strings += Str  # 把当前遍历到的内容给Str
        for num in re.compile('\d+').finditer(self.test_strings):
            self.NumList.append((int(num.group()), num.span()[0]))
        return [self.converted_strings.replace("【·&……】", "什"), sorted(self.NumList, key=lambda x: x[1])]
