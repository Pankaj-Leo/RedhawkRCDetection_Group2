import serial
import time
import serial.tools.list_ports

def find_arduino_port():
    """Scan for an Arduino connected and return its port."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if 'Arduino' in p.description or 'arduino' in p.description:
            print(f"Arduino found on port {p.device}")
            return p.device
    return None

def send_command(ser, steering, velocity):
    """Send a steering and velocity command to the Arduino."""
    command = f"{steering},{velocity}\n"
    try:
        ser.write(command.encode('utf-8'))
    except serial.SerialException as e:
        print(f"Error sending command: {e}")

def main():
    arduino_port = find_arduino_port()

    if arduino_port is None:
        print("Arduino not found. Please check your connection.")
        return

    try:
        with serial.Serial(arduino_port, 9600, timeout=1) as ser:
            print(f"Connected to Arduino on {arduino_port}")
            time.sleep(2)  # give some time for the Arduino to reset

            counter = 1
            while True:
                print("Test: ", counter)
                # Forward
                send_command(ser, 90, 100)
                print("Forward")
                time.sleep(2)

                # Turn left
                send_command(ser, 45, 100)
                print("Turn Left")
                time.sleep(2)

                # Turn right
                send_command(ser, 135, 100)
                print("Turn Right")
                time.sleep(2)

                # Stop
                send_command(ser, 90, 0)
                print("Stop")
                time.sleep(2)

                counter += 1

    except serial.SerialException as e:
        print(f"Serial communication error: {e}")
    except KeyboardInterrupt:
        print("Program terminated by user.")
    finally:
        if 'ser' in locals() and ser.is_open:
            send_command(ser, 90, 0)  # send stop command
            ser.close()
            print("Serial connection closed.")

if __name__ == "__main__":
    main()


cap.release()
cv2.destroyAllWindows()
