import cv2
from keras.models import load_model
import numpy as np
from collections import deque
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class DevReg:
    def webCam(self):
        model = load_model('devnagri.h5')
        print(model)

        letter_count = {0: 'CHECK', 1:'01_ka', 2:'02_kha', 3:'03_ga', 4: '04_gha', 5:'05_kna', 6:'06_cha',
                        7:'07_chha', 8:'08_ja', 9:'09_jha', 10:'10_yna',
                        11: '11_tta(Tamatar)', 12: '12_ttha', 13: '13_dda', 14: '14_ddha', 15: '15_adna', 16: '16_ta(Tabla)', 17: '17_tha',
                        18: '18_da', 19: '19_dha', 20: '20_na', 21: '21_pa', 22: '22_pha', 23: '23_ba', 24: '24_bha', 25: '25_ma',
                        26: '26_yaw(yash)',
                        27: '27_ra', 28: '28_la', 29: '29_va (veer)',30: '30_sha (Shalgam)',
                        31: '31_pa', 32: '32_sa', 33: '33_ha', 34: '34_chhya', 35: '35:tra', 36: '36_gya',37: 'CHECK'}
        """,37:'0',
                     38:'1',
                     39:'2',
                     40:'3',
                     41:'4',
                     42:'5',
                     43:'6',
                     44:'7',
                     45:'8',
                     46:'9', 47: 'CHECK'}"""

        def keras_predict(model, image):
            processed = keras_process_image(image)
            print("processed: "+str(processed.shape))
            pred_probab = model.predict(processed)[0]
            pred_class = list(pred_probab).index(max(pred_probab))
            return max(pred_probab), pred_class

        def keras_process_image(img):
            image_x = 32
            image_y = 32
            img = cv2.resize(img,(image_x,image_y))
            img = np.array(img, dtype = np.float32)
            img = np.reshape(img, (-1, image_x, image_y, 1))
            print(img)
            type(img)
            return img

        try:
            cap = cv2.VideoCapture(0)
            Lower_blue = np.array([110, 50, 50])
            Upper_blue = np.array([130, 255, 255])
            pred_class = 0
            pts = deque(maxlen=5120000)
            blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
            digit = np.zeros((200, 200, 3), dtype=np.uint8)
            while cap.isOpened():
                ret, img = cap.read()
                img = cv2.flip(img, 1)
                imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(imgHSV, Lower_blue, Upper_blue)
                blur = cv2.medianBlur(mask, 15)
                blur = cv2.GaussianBlur(blur, (5, 5), 0)
                thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
                center = None
                if len(cnts) >= 1:
                    contour = max(cnts, key=cv2.contourArea)
                    if cv2.contourArea(contour) > 250:
                        ((x, y), radius) = cv2.minEnclosingCircle(contour)
                        cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                        cv2.circle(img, center, 5, (0, 0, 255), -1)
                        M = cv2.moments(contour)
                        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                        pts.appendleft(center)
                        for i in range(1, len(pts)):
                            if pts[i - 1] is None or pts[i] is None:
                                continue
                            cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 10)
                            cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 5)
                elif len(cnts) == 0:
                    if len(pts) != []:
                        blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
                        blur1 = cv2.medianBlur(blackboard_gray, 15)
                        blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                        thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                        blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
                        if len(blackboard_cnts) >= 1:
                            cnt = max(blackboard_cnts, key=cv2.contourArea)
                            print(cv2.contourArea(cnt))
                            if cv2.contourArea(cnt) > 2000:
                                x, y, w, h = cv2.boundingRect(cnt)
                                digit = blackboard_gray[y:y + h, x:x + w]
                                # new Image = process_letter(digit)
                                pred_probab, pred_class = keras_predict(model, digit)
                                print(pred_class, pred_probab)

                    pts = deque(maxlen=512)
                    blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(img, "Conv Network: " + str(letter_count[pred_class]), (10, 470),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Frame", img)
                cv2.imshow("Contour", thresh)
                k = cv2.waitKey(10)
                if k ==27:
                    break

        except cv2.error as e:
            pass

    def canvasReg(self):
        model = load_model('devnagri.h5')
        print(model)

        letter_count = {0: 'CHECK', 1: '01_ka', 2: '02_kha', 3: '03_ga', 4: '04_gha', 5: '05_kna', 6: '06_cha',
                        7: '07_chha', 8: '08_ja', 9: '09_jha', 10: '10_yna',
                        11: '11_tta(Tamatar)', 12: '12_ttha', 13: '13_dda', 14: '14_ddha', 15: '15_adna',
                        16: '16_ta(Tabla)', 17: '17_ttha',
                        18: '18_da', 19: '19_dha', 20: '20_na', 21: '21_pa', 22: '22_pha', 23: '23_ba', 24: '24_bha',
                        25: '25_ma',
                        26: '26_yaw(yash)',
                        27: '27_ra', 28: '28_la', 29: '29_waw(veer)', 30: '30_sha (Shalgam)',
                        31: '31_pa', 32: '32_sa', 33: '33_ha', 34: '34_ga', 35: '35:tra', 36: '36_gya',
                        37: 'CHECK'}
        """,37:'0',
                     38:'1',
                     39:'2',
                     40:'3',
                     41:'4',
                     42:'5',
                     43:'6',
                     44:'7',
                     45:'8',
                     46:'9', 47: 'CHECK'}"""

        def keras_predict(model, image):
            processed = keras_process_image(image)
            print("processed: " + str(processed.shape))
            pred_probab = model.predict(processed)[0]
            pred_class = list(pred_probab).index(max(pred_probab))
            print("KProcess2")
            return max(pred_probab), pred_class

        def keras_process_image(img):
            image_x = 32
            image_y = 32
            img = cv2.resize(img, (image_x, image_y))
            img = np.array(img, dtype=np.float32)
            img = np.reshape(img, (-1, image_x, image_y, 1))
            print("KProcess")
            print(img)
            return img

        pred_class = 0
        rgb = cv2.imread("./temp.png")
        gray =  cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        #binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        #img = keras_process_image(gray)

        pred_probab, pred_class = keras_predict(model, gray)
        print(pred_class, pred_probab)

        print("Index got")
        print(letter_count[pred_class])

        cv2.waitKey(10000)
