*Smart Animal Intrusion Detection and Deterrent System (Indoor Prototype)*

_Team members_

Kushagra Gupta – 2024MCB1299
Mayank Kumar – 2024MCB1304
Rushi Limbachiya – 2024MCB1301
Manish Kumar Chatakunda – 2024MCB1302
Saurabh Sharma – 2024MCB1311

1. Introduction to the Problem
 
Animal intrusions into farm lands lead to serious agricultural losses and interfere with farming activities.
Current preventions like scarecrows, fencing, and hand surveillance are inefficient, lacks flexibility, and are
unable to react dynamically depending on the kind of animals. Therefore, we require an intelligent, automated
system that can detect and deter animals efficiently.

2. Background and Contex
  
Recent advances in machine learning and the Internet of Things (IoT) make it possible to integrate visual
detection with automatic control mechanisms. This type of integration enables smart decision-making and
effective deterring. Outdoor testing, though, is fairly complicated owing to environmental factors and security
issues. As such, we will be creating an indoor prototype that mimics animal intrusion via projected pictures
offers a secure and efficient means of proving the potential of the system.

3. Project Objectives and Aims
   
→ The goal was to design a working indoor prototype for identification of different species of animals from
live camera images.
→ Automate effective deterrent reactions (LEDs, ultrasonic Repeller, water spray) according to recognized
animals.
→ To showcase edge-based AI inference with the IoT hardware.
→ To evaluate the system's performance, accuracy, and responsiveness under various simulated scenarios.

4. Components Used
   
1. Raspberry Pi 3 (1GB)
2.Raspberry Pi Camera Module -5MP
3.PIR Motion Sensor
4. MAX 4466 Microphone Module
5. LED Strobes, Buzzers
6. MCP-3008 Analog to Digital Converter
7. Jumper Wires
8. Breadboard
9. Optional (items issued but not used): GSM Module/Wi-Fi Module, LoRa Module

5. Workload Distribution
   
Mayank and Kushagra: Hardware installation and circuit integration, such as power regulation, sensor cabling.
Rushi and Manish: Software development and control logic in Python, handling sensor inputs .
Rushi, Manish and Saurabh: Development of machine learning model — dataset preparation, training, and
optimization for real-time identification.
Kushagra and Saurabh: System integration, synchronization of hardware and ML components, and end-to-end
debugging.
Kushagra: System performance evaluation, and report preparation and presentation materials.
All members contributed as a team to documentation, project demonstration, and troubleshooting at each
stage to provide continuous and balanced involvement.

6. Design and Implementation: -
   
-> The whole project works on raspberry pi 3, using a 32-bit OS. The ML model is integrated in the pi and
runs locally.
-> We are using a raspberry pi camera module (5MP) for live image feed, MAX4466 microphone module
for live audio detection and PIR motion sensor as well for triggering the system.
-> The ML model processes the live image clicked by the camera and checks for any animal which can
harm the crops. If it detects one, it appropriately turns on respective mitigation device, preprogrammed to run for the specific animal.
-> We have programmed the following mitigation devices according to the animal class detected as
given here: -

1. Wild Animals (wild boar, tiger, lion etc.) – Ground level traps/sirens/ drone flight
2. Stray animals (rabbits, rodents, monkey etc.) – Ultrasonic repellents/ sprinklers/ flashing
lights/ alarms sound.
3. Birds – loudspeakers with distress calls
-> All these mitigation devices are being simulated using LEDs and Buzzers.

7. Challenges in Addressing the Problem
   
-> The primary difficulties in designing such a system are to attain high detection precision with lowresource machine learning models while having the reliable control of hardware with low latency,
efficient fusion of multiple sensor inputs and stable power sharing in an indoor setting.
-> The main challenges we faced during the making of this project are listed below: -
-> Computational power of raspberry pi 3 was very low, due to which we had to shift to a low-resource
ML model which uses TensorFlow lite.
-> The camera module stopped working and thus we couldn’t integrate it with the software for a long
time.
-> The microphone provided (MAX4466) did not have enough sampling rate to process and classify the
audio inputs into different animal classes.

8. Conclusion
    
The Smart Animal Intrusion Detection and Deterrent System illustrate the successful implementation of AI and
IoT towards agricultural innovation. The indoor prototype we have developed is a controlled, safe, and
sensible model to illustrate how smart systems are able to recognize and deter animals on their own. This
project not only proves the viability of AI-based protection for crops but also sets up a scalable model for realworld applications.
