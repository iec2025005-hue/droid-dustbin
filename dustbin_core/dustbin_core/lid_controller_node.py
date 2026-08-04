#!/usr/bin/env python3
import time
import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

try:
    import RPi.GPIO as GPIO
except ImportError:
    pass

SERVO_GPIO_PIN = 18
SERVO_PWM_FREQ = 50
SERVO_DUTY_CLOSED = 5.0
SERVO_DUTY_OPEN   = 7.5
SERVO_SWEEP_STEPS = 20
SERVO_STEP_DELAY  = 0.015

def init_servo():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(SERVO_GPIO_PIN, GPIO.OUT)
    pwm = GPIO.PWM(SERVO_GPIO_PIN, SERVO_PWM_FREQ)
    pwm.start(SERVO_DUTY_CLOSED)
    time.sleep(0.3)
    return pwm

def servo_open(pwm):
    step_size = (SERVO_DUTY_OPEN - SERVO_DUTY_CLOSED) / SERVO_SWEEP_STEPS
    duty = SERVO_DUTY_CLOSED
    for _ in range(SERVO_SWEEP_STEPS):
        duty += step_size
        pwm.ChangeDutyCycle(duty)
        time.sleep(SERVO_STEP_DELAY)
    pwm.ChangeDutyCycle(SERVO_DUTY_OPEN)

def servo_close(pwm):
    step_size = (SERVO_DUTY_OPEN - SERVO_DUTY_CLOSED) / SERVO_SWEEP_STEPS
    duty = SERVO_DUTY_OPEN
    for _ in range(SERVO_SWEEP_STEPS):
        duty -= step_size
        pwm.ChangeDutyCycle(duty)
        time.sleep(SERVO_STEP_DELAY)
    pwm.ChangeDutyCycle(SERVO_DUTY_CLOSED)

class LidControllerNode(Node):
    def __init__(self):
        super().__init__('lid_controller_node')
        
        self.get_logger().info("Initialising Servo Hardware...")
        self.servo_pwm = init_servo()
        self.is_open = False
        
        self.subscription = self.create_subscription(
            Bool,
            '/dustbin/arm_raised',
            self.listener_callback,
            10
        )
        self.get_logger().info("Lid Controller Node Started - Waiting for /dustbin/arm_raised")

    def listener_callback(self, msg):
        want_open = msg.data
        if want_open and not self.is_open:
            self.get_logger().info("Arm raised! Opening lid...")
            servo_open(self.servo_pwm)
            self.is_open = True
        elif not want_open and self.is_open:
            self.get_logger().info("Arm lowered. Closing lid...")
            servo_close(self.servo_pwm)
            self.is_open = False

    def cleanup(self):
        self.get_logger().info("Cleaning up GPIO...")
        try:
            self.servo_pwm.ChangeDutyCycle(SERVO_DUTY_CLOSED)
            time.sleep(0.4)
            self.servo_pwm.stop()
        except Exception:
            pass
        GPIO.cleanup()

def main(args=None):
    rclpy.init(args=args)
    node = LidControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
