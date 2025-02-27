import cv2
import numpy as np
import os

# Camera parameters
cam_matrix = np.array([
    [544.060, 0, 321.091],
    [0, 544.308, 232.751],
    [0, 0, 1]
])

dist_coeff = np.array([[0.193, -0.6107, -0.0052, 0.00027, 0.59749]])
fx = cam_matrix[0, 0]
fy = cam_matrix[1, 1]
cx = cam_matrix[0, 2]
cy = cam_matrix[1, 2]
z = 445  # mm

# Create 'captures' directory if it doesn't exist
save_dir = "yolo_dataset"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize image counter
image_count = 1

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # Undistortion
    img2 = cv2.undistort(img, cam_matrix, dist_coeff)

    # Show the frame
    cv2.imshow("Inspection", img2)

    key = cv2.waitKey(1) & 0xFF

    # If "s" key is pressed, capture and save the image with a unique filename
    if key == ord('s'):
        # Create a black image of size 160x640 (to pad at the bottom)
        black_pad = np.zeros((160, 640, 3), dtype=np.uint8)

        # Stack the original image and black padding
        final_img = np.vstack((img2, black_pad))

        # Generate filename with counter
        filename = os.path.join(save_dir, f"captured_image_{image_count}.png")

        # Save the image
        cv2.imwrite(filename, final_img)
        print(f"Image {image_count} captured and saved as '{filename}'")

        # Increment counter
        image_count += 1

    # Exit if "q" is pressed
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
