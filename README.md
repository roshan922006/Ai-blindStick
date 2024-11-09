# Ai-blindStick
import cv2
import numpy as np
import pygame
import RPi.GPIO as GPIO
import time
import tensorflow as tf

# Initialize Pygame for sound feedback
pygame.init()
pygame.mixer.init()

# Load pre-trained MobileNet SSD model
model = tf.saved_model.load('path_to_model_directory')  # Add your model's path here
category_index = {1: 'Person', 2: 'Car', 3: 'Bicycle'}  # Simplified class labels

# Ultrasonic sensor setup
TRIG = 23  # GPIO pin for Trigger
ECHO = 24  # GPIO pin for Echo
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# Function to play sound
def play_sound(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

# Function to measure distance with ultrasonic sensor
def measure_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    
    start_time = time.time()
    stop_time = time.time()
    
    while GPIO.input(ECHO) == 0:
        start_time = time.time()
    
    while GPIO.input(ECHO) == 1:
        stop_time = time.time()
    
    elapsed_time = stop_time - start_time
    distance = (elapsed_time * 34300) / 2
    return distance

# Function to perform object detection
def detect_objects(frame):
    input_tensor = tf.convert_to_tensor(frame)
    detections = model(input_tensor)
    return detections

# Main function
def main():
    cap = cv2.VideoCapture(0)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Object detection
            detections = detect_objects(frame)

            # Display detected objects and provide feedback
            for detection in detections:
                score = detection['score']
                if score > 0.5:  # Confidence threshold
                    label = category_index.get(detection['class_id'], 'Unknown')
                    print(f"Detected: {label} with confidence {score:.2f}")
                    
                    # Play sound based on detected object
                    if label == 'Person':
                        play_sound('person_detected.mp3')
                    elif label == 'Car':
                        play_sound('car_detected.mp3')

            # Obstacle detection with ultrasonic sensor
            distance = measure_distance()
            if distance < 50:  # Threshold distance in cm
                print("Obstacle detected!")
                play_sound('obstacle_detected.mp3')
            
            # Display the frame
            cv2.imshow('Blind Stick AI', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Program stopped")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
