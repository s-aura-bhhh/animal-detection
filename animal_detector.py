import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image
import time
import os
import argparse
import requests
import RPi.GPIO as GPIO

class AnimalDetector:
    def __init__(self, model_path):
        """
        Initialize Animal Detector with LED GPIO setup
        """
        # Initialize GPIO for LED control
        self.setup_gpio()
        
        # Define animal classes and their GPIO mappings
        self.animal_classes = self.define_animal_classes()
        self.class_names = self.load_imagenet_classes()
        
        # Load TFLite model
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        input_shape = self.input_details[0]['shape']
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        
        print(f"🦁 Animal Detector with LED Control Initialized!")
        print(f"📁 Model: {os.path.basename(model_path)}")
        print(f"🖼️ Input shape: {input_shape}")
        print(f"💡 LED Pins: 12,13,20,21,22,23 (1-second timeout)")
        print("-" * 60)

    def setup_gpio(self):
        """Initialize GPIO pins for LED control"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)  # Disable warnings
        
        self.led_pins = {
            'class1': [12,20,21,22,23],  # Dangerous predators
            'class2': [13],  # Large herbivores
            'class3': [20],  # Livestock/working animals
            'class4': [21],  # Pets (dogs/cats)
            'class5': [22],  # Birds
            'class6': [23]   # Small animals/reptiles
        }
        
        # Setup all pins as output and turn off initially
        for class_name, pin in self.led_pins.items():
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
            print(f"✅ GPIO {pin} initialized for {class_name}")
        
        print("✅ All GPIO pins initialized - All LEDs OFF")

    def define_animal_classes(self):
        """
        Define animal classes and their GPIO mappings
        """
        return {
            'class1': {  # Dangerous predators - GPIO 12
                'tiger', 'lion', 'wild boar', 'hyena', 'leopard', 'cheetah', 'jaguar',
                'snow leopard', 'cougar', 'lynx', 'brown bear', 'american black bear',
                'ice bear', 'sloth bear', 'wolf', 'coyote', 'dingo', 'dhole'
            },
            'class2': {  # Large herbivores - GPIO 13
                'elephant', 'hippopotamus', 'zebra', 'giraffe', 'rhinoceros',
                'indian elephant', 'african elephant'
            },
            'class3': {  # Livestock/working animals - GPIO 20
                'hog', 'water buffalo', 'bison', 'ram', 'bighorn', 'ibex', 'ox',
                'cow', 'bull', 'staffordshire bullterrier', 'bull mastiff'
            },
            'class4': {  # Pets (dogs/cats) - GPIO 21
                # All dogs except staffordshire bullterrier
                'chihuahua', 'japanese spaniel', 'maltese dog', 'pekinese', 'shih-tzu',
                'blenheim spaniel', 'papillon', 'toy terrier', 'rhodesian ridgeback',
                'afghan hound', 'basset hound', 'beagle', 'bloodhound', 'bluetick',
                'black-and-tan coonhound', 'walker hound', 'english foxhound',
                'redbone', 'borzoi', 'irish wolfhound', 'italian greyhound', 'whippet',
                'ibizan hound', 'norwegian elkhound', 'otterhound', 'saluki',
                'scottish deerhound', 'weimaraner', 'american staffordshire terrier',
                'bedlington terrier', 'border terrier', 'kerry blue terrier',
                'irish terrier', 'norfolk terrier', 'norwich terrier', 'yorkshire terrier',
                'wire-haired fox terrier', 'lakeland terrier', 'sealyham terrier',
                'airedale', 'cairn', 'australian terrier', 'dandie dinmont', 'boston bull',
                'miniature schnauzer', 'giant schnauzer', 'standard schnauzer',
                'scotch terrier', 'tibetan terrier', 'silky terrier',
                'soft-coated wheaten terrier', 'west highland white terrier', 'lhasa',
                'flat-coated retriever', 'curly-coated retriever', 'golden retriever',
                'labrador retriever', 'chesapeake bay retriever', 'german short-haired pointer',
                'vizsla', 'english setter', 'irish setter', 'gordon setter', 'brittany spaniel',
                'clumber', 'english springer', 'welsh springer spaniel', 'cocker spaniel',
                'sussex spaniel', 'irish water spaniel', 'kuvasz', 'schipperke', 'groenendael',
                'malinois', 'briard', 'kelpie', 'komondor', 'old english sheepdog',
                'shetland sheepdog', 'collie', 'border collie', 'bouvier des flandres',
                'rottweiler', 'german shepherd', 'doberman', 'miniature pinscher',
                'greater swiss mountain dog', 'bernese mountain dog', 'appenzeller',
                'entlebucher', 'boxer', 'tibetan mastiff', 'french bulldog', 'great dane',
                'saint bernard', 'eskimo dog', 'malamute', 'siberian husky', 'dalmatian',
                'affenpinscher', 'basenji', 'pug', 'leonberg', 'newfoundland',
                'great pyrenees', 'samoyed', 'pomeranian', 'chow', 'keeshond',
                'brabancon griffon', 'pembroke', 'cardigan', 'toy poodle',
                'miniature poodle', 'standard poodle', 'mexican hairless',
                # All cats
                'tabby', 'tiger cat', 'persian cat', 'siamese cat', 'egyptian cat'
            },
            'class5': {  # Birds - GPIO 22
                'cock', 'hen', 'ostrich', 'brambling', 'goldfinch', 'house finch',
                'junco', 'indigo bunting', 'robin', 'bulbul', 'jay', 'magpie',
                'chickadee', 'water ouzel', 'kite', 'bald eagle', 'vulture',
                'great grey owl', 'peacock', 'quail', 'partridge', 'african grey',
                'macaw', 'sulphur-crested cockatoo', 'lorikeet', 'coucal', 'bee eater',
                'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake',
                'red-breasted merganser', 'goose', 'black swan', 'white stork',
                'black stork', 'spoonbill', 'flamingo', 'little blue heron',
                'great egret', 'bittern', 'crane', 'limpkin', 'common gallinule',
                'american coot', 'bustard', 'ruddy turnstone', 'red-backed sandpiper',
                'redshank', 'dowitcher', 'oystercatcher', 'pelican', 'king penguin',
                'albatross'
            },
            'class6': {  # Small animals/reptiles - GPIO 23
                'hamster', 'porcupine', 'fox squirrel', 'marmot', 'beaver', 'guinea pig',
                'sorrel', 'wood rabbit', 'hare', 'angora rabbit', 'rat', 'mouse',
                'meerkat', 'mongoose', 'weasel', 'mink', 'polecat', 'black-footed ferret',
                'otter', 'skunk', 'badger', 'armadillo', 'three-toed sloth',
                # Reptiles and amphibians
                'axolotl', 'nematode', 'fire salamander', 'smooth newt', 'newt',
                'spotted salamander', 'bullfrog', 'tree frog', 'tailed frog',
                'loggerhead turtle', 'leatherback turtle', 'mud turtle', 'terrapin',
                'box turtle', 'banded gecko', 'common iguana', 'american chameleon',
                'whiptail lizard', 'agama', 'frilled lizard', 'alligator lizard',
                'gila monster', 'green lizard', 'african chameleon', 'komodo dragon',
                'african crocodile', 'american alligator', 'thunder snake',
                'ringneck snake', 'hognose snake', 'green snake', 'king snake',
                'garter snake', 'water snake', 'vine snake', 'night snake',
                'boa constrictor', 'rock python', 'indian cobra', 'green mamba',
                'sea snake', 'horned viper', 'diamondback rattlesnake', 'sidewinder'
            }
        }

    def load_imagenet_classes(self):
        """
        Load official ImageNet class labels
        """
        imagenet_labels_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
        local_labels_path = "imagenet_classes.txt"
        
        if not os.path.exists(local_labels_path):
            print("📥 Downloading ImageNet class labels...")
            try:
                response = requests.get(imagenet_labels_url)
                with open(local_labels_path, 'w') as f:
                    f.write(response.text)
                print("✅ ImageNet labels downloaded successfully!")
            except Exception as e:
                print(f"❌ Could not download labels: {e}")
                return self.get_fallback_classes()
        
        # Read the class labels
        with open(local_labels_path, 'r') as f:
            class_names = f.read().splitlines()
        
        # Remove 'background' class if present
        if class_names and class_names[0] == 'background':
            class_names = class_names[1:]
        
        return class_names

    def get_fallback_classes(self):
        """Fallback class list"""
        print("⚠️ Using fallback class list")
        return [f"class_{i}" for i in range(1000)]

    def get_animal_class_for_prediction(self, animal_name):
        """
        Determine which class the detected animal belongs to
        """
        animal_name_lower = animal_name.lower()
        
        for class_name, animals in self.animal_classes.items():
            if animal_name_lower in animals:
                return class_name
        
        return None

    def control_leds(self, animal_class):
        try:
                # Turn off all LEDs first
                self.turn_off_all_leds()
        
                # Turn on the specific LED for the detected class
                if animal_class in self.led_pins:
                        for led in self.led_pins[animal_class]:
                                target_pin = led
                                GPIO.output(target_pin, GPIO.HIGH)
                                print(f"LED {target_pin} ON - Class: {animal_class}")
            
            # Keep LED on for 1 second
                        print("Waiting 1 second...")
                        time.sleep(1)
            
            # Turn off the LED after 1 second
                        for led in self.led_pins[animal_class]:
                                target_pin = led
                                GPIO.output(target_pin, GPIO.LOW)
                                print(f"LED {target_pin} OFF after 1 second")
                else:
                        print("No LED activated - Unknown animal class")
            
        except Exception as e:
                print(f"Error controlling LEDs: {e}")

    def turn_off_all_leds(self):
        """Turn off all LED pins"""
        for pin in self.led_pins.values():
            GPIO.output(pin, GPIO.LOW)

    def preprocess_image(self, image_path):
        """
        Preprocess image for MobileNetV2
        """
        try:
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            print(f"📷 Loaded image: {os.path.basename(image_path)}")
            print(f"📐 Original size: {original_size[0]}x{original_size[1]}")
            
            # Resize image to model input size
            image = image.resize((self.input_width, self.input_height))
            print(f"🔄 Resized to: {self.input_width}x{self.input_height}")
            
            # Convert to numpy array
            image_array = np.array(image, dtype=np.float32)
            
            # MobileNetV2 preprocessing: scale pixels to [-1, 1]
            image_array = (image_array / 127.5) - 1.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None

    def predict(self, image_path, top_k=5):
        """
        Run inference and filter for animal classes only
        """
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None
        
        # Preprocess image
        input_data = self.preprocess_image(image_path)
        if input_data is None:
            return None
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        
        # Run inference
        print("🤖 Running inference...")
        start_time = time.time()
        self.interpreter.invoke()
        inference_time = time.time() - start_time
        
        # Get output tensor
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        all_predictions = output_data[0]
        
        # Consider only first 397 classes (animals in ImageNet)
        animal_predictions = []
        for idx in range(397):  # First 397 classes are animals
            if idx < len(all_predictions):
                confidence = float(all_predictions[idx])
                if idx < len(self.class_names):
                    class_name = self.class_names[idx]
                    animal_predictions.append((class_name, confidence, idx))
        
        # Sort by confidence and get top K
        animal_predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = animal_predictions[:top_k]
        
        print(f"⚡ Inference time: {inference_time*1000:.2f} ms")
        print("-" * 60)
        
        return top_predictions, inference_time

    def print_results(self, predictions, image_path):
        """
        Print formatted results and control LEDs
        """
        if not predictions:
            print("❌ No animal detected in the image")
            return None
        
        print(f"🎯 ANIMAL DETECTION RESULTS for {os.path.basename(image_path)}")
        print("=" * 70)
        
        top_animal_class = None
        
        for i, (animal, confidence, idx) in enumerate(predictions, 1):
            confidence_percent = confidence * 100
            bar_length = int(confidence * 25)
            bar = "█" * bar_length + "░" * (25 - bar_length)
            
            # Determine animal class
            animal_class = self.get_animal_class_for_prediction(animal)
            class_indicator = f"[{animal_class}]" if animal_class else "[Unknown]"
            
            if i == 1:  # Top prediction
                top_animal_class = animal_class
                print(f"🏆 {i:2d}. {animal:25s} {class_indicator:12s} {confidence_percent:5.1f}% [{bar}]")
            else:
                print(f"   {i:2d}. {animal:25s} {class_indicator:12s} {confidence_percent:5.1f}% [{bar}]")
        
        print("=" * 70)
        
        # Control LEDs based on top prediction with 1-second timeout
        if top_animal_class:
            self.control_leds(top_animal_class)
            top_animal, top_confidence, top_idx = predictions[0]
            print(f"🔍 Top detection: {top_animal} ({top_confidence*100:.1f}% confidence)")
            print(f"💡 Activated: {top_animal_class} -> GPIO {self.led_pins.get(top_animal_class, 'N/A')} for 1 second")
        else:
            print("💡 No LED activated - Unknown animal class")
        
        return top_animal_class

    def cleanup(self):
        """Cleanup GPIO"""
        try:
            self.turn_off_all_leds()
            GPIO.cleanup()
            print("✅ GPIO cleanup completed")
        except Exception as e:
            print(f"❌ Error during GPIO cleanup: {e}")

def main():
    parser = argparse.ArgumentParser(description='Animal Detector with LED Control')
    parser.add_argument('--model', type=str, default='mobilenetv2_imagenet.tflite',
                       help='Path to TFLite model file')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image file')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to show (default: 5)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("Please make sure the MobileNetV2 TFLite model is in the current directory")
        return
    
    detector = None
    try:
        detector = AnimalDetector(args.model)
        results, inference_time = detector.predict(args.image, args.top_k)
        
        if results:
            detector.print_results(results, args.image)
        else:
            print("❌ No animal detections found")
            
    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if detector:
            detector.cleanup()

if __name__ == "__main__":
    main()
