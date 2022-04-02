from control import gpio
import time

m = None


m=gpio.Mecanum_wheel()
m.uart_init()
m.car_speed["car_go"]=500
m.car_go()
time.sleep(2)
m.car_speed["car_back"]=500
m.car_back()
time.sleep(2)
m.car_speed["car_across_l"]=500
m.car_across_l()
time.sleep(2)
m.car_speed["car_across_r"]=500
m.car_across_r()
time.sleep(2)
m.car_speed["car_turn_l"]=50
m.car_turn_l()
time.sleep(2)
m.car_speed["car_turn_r"]=50
m.car_turn_r()
time.sleep(2)
m.stop()