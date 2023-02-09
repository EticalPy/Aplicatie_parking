# Importam librariile
import numpy as np
import torch
import cv2
# import easyocr
import numpy

# Citim modelul rezultat dupa procesul de invatare

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='/media/bogdan/SSD2/PROIECTE2023/detectare_obiecte_yolov5_masca/model1/best.pt',
                       force_reload=True)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    detector = model(frame)

    info = detector.pandas().xyxy[0].to_dict(orient="records")
    # print(info)

    cv2.imshow('Detector_numere', np.squeeze(detector.render()))

    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()