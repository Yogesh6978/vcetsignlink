import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the pre-trained model
model_dict = pickle.load(open(r'C:\\Users\\kavya\\Desktop\\sign-language-detector-python-master\\model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels dictionary
labels_dict = {0: 'HI', 1: 'HOW ARE YOU', 2: 'NEED WATER',3: 'HUNGRY'}

# Initialize the camera (default index 0)
cap = cv2.VideoCapture(0)  # Use 0 for the default camera. Change if you have multiple cameras.

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    data_aux = []
    x_ = []
    y_ = []

    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Get frame dimensions
    H, W, _ = frame.shape

    # Convert the frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Draw hand landmarks on the frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # Image to draw
                hand_landmarks,  # Model output
                mp_hands.HAND_CONNECTIONS,  # Hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Extract hand landmarks and normalize them
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Normalize landmarks (x, y)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Debug: Print the input data
      #  print("Input Data to Model:", data_aux)

        # Define bounding box for the hand
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Predict the hand gesture using the trained model
        prediction = model.predict([np.asarray(data_aux)])

        # Get the predicted character label
        predicted_character = labels_dict[int(prediction[0])]

        # Debug: Print the predicted character
        print("Predicted Character:", predicted_character)

        # Draw bounding box and predicted character on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    # Show the frame with hand landmarks and prediction
    cv2.imshow('Frame', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
