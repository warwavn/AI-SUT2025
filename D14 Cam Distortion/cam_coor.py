import cv2 
from ultralytics import YOLO   
import numpy as np

cam_matrix = np.array([
    [547.943,   0.0,        340.749,],
    [0.0,       547.249,    242.417,],
    [0,         0,          1]
])

dist_coeff = np.array([0.228,-0.832,-0.00070,0.0064,0.8576])

fx = cam_matrix[0,0]
fy = cam_matrix[1,1]
cx = cam_matrix[0,2]
cy = cam_matrix[1,2]
z = 422                      # distance in mm

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)          #if run with another matrix MUST review/change

model = YOLO("yolo11n.pt")

while cap.isOpened():
    ret, img = cap.read()
    img_un = cv2.undistort(img, cam_matrix,dist_coeff)               #undistortion img

    results = model.predict(img_un)

    for res in results:
        boxes = res.boxes.xyxy.cpu().numpy()                            #boxes, masks, probs; for cpu
        classes = res.boxes.cls.cpu().numpy()                           #ex. this kind of defect then area?
        for box,cls_id in zip(boxes,classes):
            b = box.tolist()
            x1, y1, x2, y2 = b
            cx = (x2+x1)/2                                      #centroid px
            cy = (y2+y1)/2

            x_robot1 = (cy/fy) * z
            y_robot1 = ((cx-320)/fx) * z                                 #320 = max pixel 640/2

            cv2.rectangle(img_un, (int(x1),int(y1)), (int(x2),int(y2)),     #cor pixel in int\
                            (255,0,0), 2)                                    #c and thickness in pixel
            cv2.circle(img_un, (int(cx),int(cy)), 5,
                            (0,0,255), -1)
             
            text = f"X: {x_robot1:.2f}, Y: {y_robot1:.2f} mm"
            cv2.putText(img_un, text, (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Inspection",img_un)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break


cap.release()
cv2.destroyAllWindows()
    # cv2.waitKey(1)