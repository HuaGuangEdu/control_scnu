a = int(input('请输入数列下标：'))
i = 1
a1 = 0
a2 = 1
while not i == a:
  a3 = a1 + a2
  a1 = a2
  a2 = a3
  i = i + 1
if i == 1:
  print(a1,end ="")
if i == 2:
  print(a2,end ="")
if i > 2:
  print(a1,end ="")