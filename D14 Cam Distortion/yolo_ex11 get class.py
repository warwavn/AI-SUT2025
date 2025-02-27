import cv2 
from ultralytics import YOLO        
#bounding box, class name, confidence level

cap = cv2.VideoCapture(r"C:\Users\User\Desktop\test\car_video.mp4")
model = YOLO("yolo11n.pt")
print(model.names)

car_count = 0
# crossed_cars = set() 

# while cap.isOpened():                                                   #check cam buffer if config the delay
#     ret, img = cap.read()
#     results = model(img,show =False)                                    #array

#     #get values from model
#     for res in results:
#         boxes = res.boxes.xyxy.cpu().numpy()                            #boxes, masks, probs; for cpu
#         classes = res.boxes.cls.cpu().numpy()                           #ex. this kind of defect then area?
#         for box,cls_id in zip(boxes,classes):
#             b = box.tolist()
#             x1, y1, x2, y2 = b

#             if cls_id == 2:                                             #from COCO dataset
#                 cx = int((x2+x1)/2)
#                 cy = int((y2+y1)/2)
#                 cv2.circle(img, (cx,cy), 10,
#                            (255,0,0), -1)                               #(B,G,R), line; -1 = filled
                
#                 ref_line = 300                                          #350 px (dnward)
#                 offset = 3

#                 if cy > ref_line - offset and cy < ref_line + offset:
#                     car_count +=1
#     cv2.line(img,(0,ref_line), (640,ref_line),(255,0,0),2)                           #coor xy; 500 can repplce with vid shape
#     cv2.putText(img, f'Total Cars: {car_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     cv2.imshow("VDO",img)
#     cv2.waitKey(1)



