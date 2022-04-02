from control import yuyin
import random
import time

s = None
a = None
b = None
c = None
name = None
audio = None
speech = None
word = None
result = None


s=yuyin.Yuyin()
while True:
  a = random.randint(1, 100)
  b = random.randint(1, 100)
  c = a + b
  print(('结果等于：' + str(c)))
  name = ''.join([str(x) for x in [b, '加', b, '等于']])
  s.tts(name,audio)
  s.play_music(audio)
  time.sleep(3)
  s.my_record(3,speech)
  word = s.stt(speech)
  result = str(c)
  if result in word:
    s.tts('你真棒',audio)
  else:
    s.tts('真可惜',audio)
  s.play_music(audio)