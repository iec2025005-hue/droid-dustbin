# Droid Dustbin

An autonomous dustbin robot built with a Raspberry Pi 4. It uses a Raspberry Pi Camera Module and Google's MediaPipe Pose detection to recognize when a person is standing in front of it and raises their arm. When an arm raise is detected, a servo motor automatically opens the dustbin lid.

## Hardware Requirements
*   Raspberry Pi 4
*   Raspberry Pi Camera Module (connected via CSI)
*   MG995 Servo Motor
*   External 5V Power Supply (for the servo)

## Software Requirements
*   Raspberry Pi OS Bookworm (64-bit)
*   Python 3.11

## Installation

1. Install system dependencies:
```bash
sudo apt update
sudo apt install -y python3-opencv python3-picamera2 python3-numpy python3-venv python3-rpi.gpio
```

2. Clone this repository:
```bash
git clone https://github.com/YOUR-USERNAME/droid-dustbin.git
cd droid-dustbin
```

3. Create a virtual environment with system site packages enabled:
```bash
python3 -m venv venv --system-site-packages
source venv/bin/activate
```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Running the Project
Activate your virtual environment and run the main script:
```bash
source venv/bin/activate
python main.py
```
