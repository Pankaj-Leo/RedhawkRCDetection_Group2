# Prototype.py

import serial
import time
from image_processing_module import ImageProcessor

def setup_serial_connection(port, baudrate=9600):
    try:
        ser = serial.Serial(port, baudrate)
        time.sleep(2)  # Allow some time for the Arduino to reset
        return ser
    except serial.SerialException as e:
        print(f"Failed to connect to Arduino on {port}: {str(e)}")
        return None

def send_command(ser, command):
    try:
        ser.write(f"{command}\n".encode('utf-8'))
    except serial.SerialException as e:
        print(f"Failed to send command {command}: {str(e)}")

def control_logic(angle, distance):
    steering = 90  # Neutral steering angle
    throttle = 90  # Neutral throttle position
    if angle < -10:
        steering = 70
    elif angle > 10:
        steering = 110
    if distance > 1000:
        throttle = 110
    elif distance < 500:
        throttle = 70
    return steering, throttle

def main():
    image_processor = ImageProcessor('Redhawk.png')
    ser = setup_serial_connection('/dev/ttyACM0')
    if ser is None:
        return

    try:
        while True:
            success, angle, distance = image_processor.process_frame()
            if not success:
                continue
            steering, throttle = control_logic(angle, distance)
            send_command(ser, f"{steering},{throttle}")
            time.sleep(0.1)
    finally:
        image_processor.release()
        if ser:
            ser.close()

if __name__ == "__main__":
    main()
