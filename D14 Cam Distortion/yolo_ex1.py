import cv2 
from ultralytics import YOLO        #ver 5,8,11

img = cv2.imread(r"C:\Users\User\Desktop\test\traffic.jpg")

model = YOLO("yolo11x.pt")

result = model(img,show=True)

# cv2.imshow("traffic",img)

cv2.waitKey(0)

