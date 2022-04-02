from control import shijue1

a = None


a=shijue1.Img()
a.camera(0)
a.face_detect_init("face")
while True:
  a.get_img()
  a.face_detect()
  a.name_windows('img')
  a.show_image('img')
  a.delay(1)
  print(a.color_data)
