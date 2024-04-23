import RPi.GPIO as gpio
import time
#gpio.setwarnings(False)
#gpio.cleanup()
gpio.setmode(gpio.BOARD)

ENB = 34
INT = 32
#gpio.setup(INT,gpio.OUT)
gpio.setup(INT,gpio.OUT)
gpio.output(INT,gpio.HIGH)
#gpio.output(INT,gpio.HIGH)
#pwm2 = gpio.PWM(INT, 2e4)
#pwm2.start(10)
time.sleep(3)
gpio.cleanup()
