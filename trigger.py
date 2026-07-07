#!/usr/bin/env python3
import time
import subprocess
import spidev
import RPi.GPIO as GPIO

# Configuration
MIC_THRESHOLD_HIGH = 650
MIC_THRESHOLD_LOW = 450
PIR_PIN = 17  # Change this to your PIR sensor GPIO pin
SPI_CHANNEL = 0  # MCP3008 channel for microphone
CAPTURE_DELAY = 2  # Seconds to wait between detections to avoid multiple triggers

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(PIR_PIN, GPIO.IN)

# Initialize SPI for MCP3008
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000

def read_adc(channel):
    """Read from MCP3008 ADC"""
    if channel < 0 or channel > 7:
        return -1

    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data

def capture_image():
    """Capture image using rpicam-still as captured.jpg with 224x224 size"""
    try:
        print("Capturing 224x224 image...")
        result = subprocess.run([
            'rpicam-still',
            '-o', 'captured.jpg',
            '--width', '224',
            '--height', '224',
            '--nopreview',
            '--timeout', '1000'  # Increased timeout to 1000ms
        ], check=True, capture_output=True, text=True)
        print("Image captured as captured.jpg (224x224)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error capturing image: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: rpicam-still not found. Make sure it's installed.")
        return False
    except Exception as e:
        print(f"Unexpected error during image capture: {e}")
        return False

def run_animal_detector():
    """Run animal detector on captured image"""
    try:
        print("Running animal detector...")
        subprocess.run([
            'python3', 'animal_detector.py',
            '--image', 'captured.jpg'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running animal detector: {e}")

def main():
    print("Starting Animal Detection Trigger System...")
    print(f"Microphone thresholds: {MIC_THRESHOLD_LOW}-{MIC_THRESHOLD_HIGH}")
    print(f"PIR sensor on GPIO: {PIR_PIN}")
    print("Press Ctrl+C to exit")

    last_trigger_time = 0

    try:
        while True:
            current_time = time.time()

            # Read microphone value
            mic_value = read_adc(SPI_CHANNEL)

            # Read PIR sensor
            pir_detected = GPIO.input(PIR_PIN)

            # Check triggers
            mic_trigger = mic_value > MIC_THRESHOLD_HIGH or mic_value < MIC_THRESHOLD_LOW
            pir_trigger = pir_detected

            # If either trigger is activated and enough time has passed since last capture
            if (mic_trigger or pir_trigger) and (current_time - last_trigger_time > CAPTURE_DELAY):
                print(f"Trigger detected - Mic: {mic_value}, PIR: {pir_detected}")

                # Capture image first
                if capture_image():
                    # Then run animal detector
                    run_animal_detector()
                    last_trigger_time = current_time
                else:
                    print("Failed to capture image, skipping animal detection")

            # Small delay to prevent excessive CPU usage
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        spi.close()
        GPIO.cleanup()

if __name__ == "__main__":
    main()