from control import shijue1

a = None


a=shijue1.Img()
a.camera(0)
a.predict_init('Jack')
while True:
  a.get_img()
  a.predict()
  a.name_windows('img')
  a.show_image('img')
  a.delay(1)
  print(a.face_name)