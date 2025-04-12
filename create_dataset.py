import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import logging

# Suppress TensorFlow logs (info, warning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Disable interactive mode in matplotlib (if you don't need interactive plots)
plt.ioff()  # Disable interactive mode

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Iterate over each class (subdirectory) in the DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    # Ensure that the current item is a directory
    if os.path.isdir(dir_path):
        # Iterate over each image file in the current directory
        for img_path in os.listdir(dir_path):
            img_path_full = os.path.join(dir_path, img_path)

            # Ensure that the file is an image (optional)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                data_aux = []
                x_ = []
                y_ = []

                # Read and process the image
                img = cv2.imread(img_path_full)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Use MediaPipe to process the image for hand landmarks
                results = hands.process(img_rgb)

                # Check if any hand landmarks are detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for i in range(len(hand_landmarks.landmark)):
                            # Get x, y coordinates of each hand landmark
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y

                            x_.append(x)
                            y_.append(y)

                        # Normalize the landmarks by subtracting the minimum x and y values
                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))

                    # Append the data and labels
                    data.append(data_aux)
                    labels.append(dir_)

# Save the collected data and labels using pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data collection complete. Saved to 'data.pickle'.")
