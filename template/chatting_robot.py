from control import yuyin

speech = None
audio = None


s = yuyin.Yuyin()
while True:
    s.my_record(3,speech)
    print(s.stt(speech))
    s.chat(s.stt(speech))
    s.tts(s.chat_ret,audio)
    s.play_music(audio)