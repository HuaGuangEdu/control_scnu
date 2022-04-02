from control import gpio
from numbers import Number

p = gpio.io2pwm(0)

duty = 10
flag = 1
p.start()
p.set_freq(50)
p.set_duty(duty)
while True:
  p.set_duty(duty)
  if flag == 1:
    duty = (duty if isinstance(duty, Number) else 0) + 1
    if duty == 98:
      flag = 0
  else:
    duty = duty - 1
    if duty == 1:
      flag = 1