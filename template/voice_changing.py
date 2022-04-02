from control import yuyin

s = None
speech = None


s=yuyin.Yuyin()
s.change_vol_spd_gender(1,3,"man")
for count in range(10):
  s.play_txt('你好，欢迎使用变声器小助手')
  s.my_record(5,speech)
  speech = s.stt(speech)
  print(speech,end ="")
  s.play_txt(speech)
