from control import yuyin

s = None
speech = None


s=yuyin.Yuyin()
s.my_record(3,speech)
print(s.stt(speech))