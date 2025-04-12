import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

# Open the video capture device (ensure you are using the correct camera index)
cap = cv2.VideoCapture(0)  # Try using 0 if 2 does not work

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Wait for the user to press "Q" to start collecting data
    done = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(25) == ord('q'):  # Wait for 'Q' to start collecting data
            done = True
            break

    if not done:
        continue  # Skip to the next class if the user never presses "Q"

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Save the captured frame as a jpg image in the respective class folder
        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        print(f"Saved: {img_path}")

        counter += 1

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
