import numpy as np
import torch
import cv2
import easyocr
import datetime
import time

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='/............/best.pt',
                       force_reload=True)
model.conf = 0.60
max_det = 1
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(0)
initial_time = time.time()
to_time = time.time()
# setam frame/secunda (FPS) recomand 5 sau 10
set_fps = 5
# Variabile pe care le folosim la calcularea FPS
prev_frame_time = 0
new_frame_time = 0
# Introducem Backgroundul si modulurile
imgBackground = cv2.imread('Resources/background2.png')
Modul1 = cv2.imread('Resources/Modes1/1.png')
Modul2 = cv2.imread('Resources/Modes1/2.png')
Modul5 = cv2.imread('Resources/Modes1/3.png')
Modul10 = cv2.imread('Resources/Modes1/4.png')
Modul20 = cv2.imread('Resources/Modes1/5.png')
# Variabile pentru abonati, clienti noi intrati si clienti iesiti
lista_numere = ['B123ZLI', 'B007BMW']
lista_clienti_noi = []
lista_clienti_iesiti = []
# Variabile care ne ajuta sa introducem pauza
counter = 0
counter2 = 0
counter3 = 0
while True:
    # Updatam timpul cu fiecare frame
    while_running = time.time()
    # Dacă timpul necesar este de 1/fps, atunci citim un cadru
    new_time = while_running - initial_time
    if new_time >= 1 / set_fps:
        ret, frame = cap1.read()
        succes, img = cap2.read()
        imgBackground2 = imgBackground.copy()
        detector = model(frame)
        info = detector.pandas().xyxy[0].to_dict(orient="records")
        imgBackground[162:162 + 480, 55:55 + 640] = img
        imgBackground[44:44 + 633, 808:808 + 414] = Modul1
        if ret:
            # Calculam FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps = str(fps)
            # print(fps)
            if len(info) != 0:
                for result in info:
                    x1 = int(result['xmin'])
                    y1 = int(result['ymin'])
                    x2 = int(result['xmax'])
                    y2 = int(result['ymax'])
                    cropped_img = frame[y1:y2, x1:x2]
                    cropped_img2 = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
                    # cv2.imshow('ox', cropped_img2)
                    detection_threshold = 0.4
                    region_threshold = 0.2
                    lungime = cropped_img.shape[1]
                    latime = cropped_img.shape[0]
                    reader = easyocr.Reader(['en'], gpu=False)
                    ocr_result = reader.readtext(cropped_img2)
                    aria = lungime * latime

                    if ocr_result[0][2] > detection_threshold:
                        for result in ocr_result:
                            lenght = np.sum(np.subtract(result[0][1], result[0][0]))
                            winght = np.sum(np.subtract(result[0][2], result[0][1]))
                            if lenght * winght / aria > region_threshold:
                                result_final = result[1].replace(" ", "")
                                result_final2 = result_final.replace("-", "")
                                if 5 < len(result_final2) < 8:
                                    cv2.putText(frame, result_final2[0:8], (150, 440), cv2.LINE_AA, 2, (0, 255, 0), 6)
                                    imgBackground2 = imgBackground.copy()
                                    Modul4 = Modul1.copy()
                                    imgBackground2[44:44 + 633, 808:808 + 414] = Modul4
                                    cv2.putText(Modul4, result_final2[0:8], (100, 150), cv2.LINE_AA, 2, (0, 255, 0), 6)

                                    if result_final2 in lista_numere:
                                        counter3 += 1
                                        if counter3 > 2:
                                            Modul3 = Modul2.copy()
                                            imgBackground2[44:44 + 633, 808:808 + 414] = Modul3
                                            cv2.putText(imgBackground2, result_final2[0:8], (860, 225), cv2.LINE_AA, 2,
                                                        (0, 255, 0), 6)
                                            if counter3 > 4:
                                                print(result_final2)
                                                input("Se ridica bariera...")
                                                counter3 = 0
                                    if result_final2 in lista_clienti_noi and result_final2 not in lista_numere:
                                        counter2 += 1
                                        Modul11 = Modul10.copy()
                                        imgBackground2[44:44 + 633, 808:808 + 414] = Modul11
                                        cv2.putText(imgBackground2, result_final2[0:8], (890, 200), cv2.LINE_AA, 2,
                                                    (0, 255, 0), 6)
                                        cv2.putText(imgBackground2,
                                                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                    (830, 590), cv2.LINE_AA, 1, (0, 255, 0), 2)
                                        cv2.putText(imgBackground2, "10 RON", (890, 400), cv2.LINE_AA, 2,
                                                    (0, 0, 255), 6)
                                        cv2.rectangle(imgBackground2, (100, 500), (640, 620), (0, 0, 0), cv2.FILLED)
                                        cv2.putText(imgBackground2, "Prezinta tichetul", (120, 580), cv2.LINE_AA, 2,
                                                    (0, 0, 255), 6)

                                        if counter2 == 2:
                                            numar_in_cauza = result_final2
                                            print(result_final2)
                                            input('Scaneaza tichetul si asteapta sa se ridice bariera....')
                                            lista_clienti_iesiti.insert(1,
                                                                        lista_clienti_noi.pop(
                                                                            lista_clienti_noi.index(numar_in_cauza)))
                                            print(lista_clienti_iesiti)
                                            lista_clienti_iesiti.remove(numar_in_cauza)
                                            counter2 = 0
                                            time.sleep(2)
                                            print("La revedere")
                                            print(lista_clienti_iesiti)
                                        time.sleep(2)
                                        print(counter2)
                                    time.sleep(1)
                                    if result_final2 not in lista_clienti_noi and 5 < len(result_final2) < 8:
                                        if result_final2 not in lista_clienti_noi and result_final2 not in lista_numere:
                                            counter += 1
                                            Modul6 = Modul5.copy()
                                            imgBackground2[44:44 + 633, 808:808 + 414] = Modul6
                                            cv2.putText(imgBackground2, result_final2[0:8], (890, 200), cv2.LINE_AA, 2,
                                                        (0, 255, 0), 6)
                                            cv2.rectangle(imgBackground2, (100, 500), (640, 620), (0, 0, 0), cv2.FILLED)
                                            cv2.putText(imgBackground2, "Ridica bonul", (180, 580), cv2.LINE_AA, 2,
                                                        (0, 0, 255), 6)
                                            cv2.putText(imgBackground2,
                                                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                        (830, 520), cv2.LINE_AA, 1,
                                                        (0, 255, 0), 2)
                                            if counter == 2:
                                                print(result_final2)
                                                input('Ia bonul si asteapta sa se ridice bariera...')
                                                lista_clienti_noi.append(result_final2)
                                                counter = 0
                                                time.sleep(2)
                                    else:
                                        counter = 0

            cv2.putText(imgBackground2, str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), (170, 700),
                        cv2.LINE_AA, 1,
                        (0, 255, 0), 3)
            cv2.imshow('Monitor', np.squeeze(detector.render()))
            cv2.imshow('Ecran interactiv', imgBackground2)
            initial_time = while_running
        else:
            total_time_of_video = while_running - to_time  # To get the total time of the video
            print(total_time_of_video)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
