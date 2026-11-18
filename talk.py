"""Type to the board; what it writes prints as it comes.

    python talk.py [/dev/ttyACM0]        (pip install pyserial)

A line is a request for the model (draw a house). "@head 3 4" shows layer 3, head 4 on the LEDs, "@head" the next
one, "@tri" the test triangle. Ctrl+C to quit.
"""
import sys
import threading

import serial

port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
s = serial.Serial(port, 115200, timeout=0.2)


def read():
    while True:
        data = s.read(4096)
        if data:
            sys.stdout.write(data.decode(errors="replace"))
            sys.stdout.flush()


threading.Thread(target=read, daemon=True).start()
print(f"connected to {port}; type a request, Enter to send, Ctrl+C to quit")
try:
    while True:
        line = input()
        if line.strip():
            s.write((line.strip() + "\n").encode("ascii", "ignore"))
except (KeyboardInterrupt, EOFError):
    print()
