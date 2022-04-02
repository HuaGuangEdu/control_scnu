from control import yuyin
import math

s = None
speech = None
a = None
b = None
s2 = None
speech1 = None
c = None
speech2 = None
speech3 = None


s=yuyin.Yuyin()
s.play_txt('你好,请选择模式1或0')
s.my_record(3,speech)
print(s.stt(speech))
if '一' in s.stt(speech):
  s.play_txt('请说出a的值')
  s.my_record(3,speech1)
  s.play_txt('请说出b的值')
  s.my_record(3,speech2)
  s.play_txt('请说出c的值')
  s.my_record(3,speech3)
  speech1 = a
  speech2 = b
  speech3 = c
else:
  a = int(input('请输入：a='))
  b = int(input('请输入：b='))
  c = int(input('请输入：c='))
s = (-1 * b + math.sqrt(b ** 2 - (4 * a) * c)) / (2 * a)
s2 = (-1 * b - math.sqrt(b ** 2 - (4 * a) * c)) / (2 * a)
print('x1=',end ="")
print(s)
print('x2=',end ="")
print(s2)