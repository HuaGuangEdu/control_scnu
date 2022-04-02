from control import shijue1

a = None


a=shijue1.Img()
a.camera(0)
while True:
  a.get_img()
  a.name_windows('img')
  a.show_image('img')
  a.img_flip()
  a.name_windows('mirror')
  a.show_image('mirror')
  a.delay(1)