import mss
import cv2
import numpy as np
import time

with mss.mss() as sct:

    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # Пример
    while True:
        start_time = time.time()

        # Захват экрана
        sct_img = sct.grab(monitor)
        img_rgb = np.array(sct_img)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('mapdevice.png', cv2.IMREAD_GRAYSCALE)
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

        cv2.imwrite('res.png', img_rgb)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"FPS: {fps:.2f}") # Выводим FPS для оценки производительности


