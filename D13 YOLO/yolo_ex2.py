import cv2 
from ultralytics import YOLO        
#bounding box, class name, confidence level

cap = cv2.VideoCapture(r"C:\Users\User\Desktop\test\car_video.mp4")
model = YOLO("yolo11n.pt")

while True:                                                 #check cam buffer if config the delay
    ret, img = cap.read()
    results = model(img,show =False)                         #array

    #get values from model
    for res in results:
        boxes = res.boxes.xyxy.cpu().numpy()                #boxes, masks, probs; for cpu
        for box in boxes:
            b = box.tolist()
            #x1, y1, x2, y2 = b
            x1 = b[0]
            y1 = b[1]
            x2 = b[2]
            y2 = b[3]
            cx = int((x2+x1)/2)
            cy = int((y2+y1)/2)
            cv2.circle(img, (cx,cy), 10,
                       (255,0,0), -1)                       #(B,G,R), line; -1 = filled
    cv2.imshow("VDO",img)
    cv2.waitKey(1)
    