import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera does not work!")
        break


    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape
    roi = frame[0:height, 0:int(width/2)]

    cv2.rectangle(frame, (0, 0), (int(width/2), height), (0, 255, 0), 2)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(contour) > 2000:
            cv2.drawContours(roi, [contour], -1, (0, 255, 0), 2)

            hull = cv2.convexHull(contour)

            cv2.drawContours(roi, [hull], -1, (0, 0, 255), 2)

            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)

            if defects is not None:
                count_defects = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])

                    a = np.linalg.norm(np.array(start) - np.array(end))
                    b = np.linalg.norm(np.array(far) - np.array(start))
                    c = np.linalg.norm(np.array(far) - np.array(end))

                    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

                    if angle <= np.pi / 2:
                        count_defects += 1
                        cv2.circle(roi, far, 5, [0, 0, 255], -1)

                if count_defects == 0:
                    cv2.putText(frame, "1 Finger", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif count_defects == 1:
                    cv2.putText(frame, "2 Fingers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif count_defects == 2:
                    cv2.putText(frame, "3 Fingers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif count_defects == 3:
                    cv2.putText(frame, "4 Fingers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif count_defects == 4:
                    cv2.putText(frame, "5 Fingers", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
