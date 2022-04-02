year = int(input('请输入年份：'))

while True:
  if year % 4 != 0:
    print('不是闰年')
    break
  if year % 100 == 0:
    if year % 400 == 0:
      print('是闰年')
      break
    else:
      print('不是闰年')
      break
  if year % 4 == 0:
    print('是闰年')
    break