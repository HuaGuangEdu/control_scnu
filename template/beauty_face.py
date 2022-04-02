from control import shijue1

a = None


a=shijue1.Img()
a.camera(0)
while True:
  a.get_img()
  a.beauty_face()
  a.name_windows('img')
  a.show_image('img')
  a.delay(1)
