# Smart Animal Intrusion Detection and Deterrent System (Indoor Prototype)

Detects animals using a camera on a Raspberry Pi, classifies them with a TensorFlow Lite model, and turns on a different LED depending on the animal type (simulating a real deterrent like sirens, sprinklers, or ultrasonic repellers).

**Demo video:** https://drive.google.com/file/d/1Gg1yMZYsT24E69Kwde6r915BNt_scm9f/view?usp=sharing

## Team

- Kushagra Gupta - 2024MCB1299
- Mayank Kumar - 2024MCB1304
- Rushi Limbachiya - 2024MCB1301
- Manish Kumar Chatakunda - 2024MCB1302
- Saurabh Sharma - 2024MCB1311

## How it works

1. `trigger.py` keeps checking a PIR motion sensor and a microphone (read through an MCP3008 ADC).
2. When either one is triggered, it captures a 224x224 image using the Pi camera and saves it as `captured.jpg`.
3. `animal_detector.py` loads the MobileNetV2 TFLite model and runs it on that image.
4. It looks at the top prediction and checks which category it belongs to.
5. Based on the category, it turns on the matching LED for 1 second, then turns it off.

## Categories and GPIO pins

| Category | Examples | GPIO pins |
|---|---|---|
| Dangerous predators | tiger, lion, wild boar, wolf | 12, 20, 21, 22, 23 |
| Large herbivores | elephant, zebra, rhino | 13 |
| Livestock/working animals | ox, bison, hog | 20 |
| Pets | dogs, cats | 21 |
| Birds | crows, waterfowl, birds of prey | 22 |
| Small animals/reptiles | rodents, rabbits, snakes, lizards | 23 |

## Tech involved

The whole thing runs locally on the Pi, no cloud calls needed for inference. We use TensorFlow Lite (through the `tflite_runtime` package) because the Pi 3 doesn't have the power to run a full TensorFlow model at a usable speed, and MobileNetV2 is a small, efficient model that still gives decent accuracy for image classification. The image is captured using `rpicam-still`, then resized and normalized with Pillow and NumPy before being fed into the model. Sensor reading is done with `RPi.GPIO` for the PIR sensor and `spidev` for talking to the MCP3008 ADC, since the Pi's GPIO pins can only read digital signals and the microphone gives an analog output. Everything is written in plain Python and runs as two scripts, one for sensing and capture, and one for classification and LED control.

## Hardware used

- Raspberry Pi 3 (1GB)
- Pi Camera Module (5MP)
- PIR motion sensor
- MAX4466 microphone
- MCP3008 ADC
- LEDs/buzzers
- Breadboard and jumper wires

## Files

- `trigger.py` - runs the sensor loop and captures images
- `animal_detector.py` - runs the model and controls the LEDs
- `imagenet_classes.txt` - class labels used by the model
- `mobilenetv2_imagenet.tflite` - the model file
- `captured.jpg` - last captured image

## Setup

```bash
sudo apt update
sudo apt install -y python3-pip rpicam-apps
pip3 install tflite-runtime numpy pillow requests RPi.GPIO spidev
```

Turn on SPI first: `sudo raspi-config` -> Interface Options -> SPI -> Enable.

Wiring:
- PIR sensor -> GPIO 17
- Microphone -> MCP3008 channel 0 -> SPI0
- LEDs -> GPIO 12, 13, 20, 21, 22, 23

## Running it

Run the full system:

```bash
python3 trigger.py
```

Press Ctrl+C to stop.

To test the model on one image directly, without the sensors:

```bash
python3 animal_detector.py --image captured.jpg
```

You can also pass `--model` (path to the tflite file) and `--top_k` (how many predictions to show).
