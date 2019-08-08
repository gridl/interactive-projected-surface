import cv2
import numpy as np
import pyautogui

def run():

    cap = cv2.VideoCapture(1)
    _, frame = cap.read()

    fromCenter = False
    r = cv2.selectROI(frame, fromCenter)

    screen_width = r[2]
    screen_height = r[3]

    x_mult = 3200 / screen_width
    y_mult = 1800 / screen_height

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, -1)
        frame = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask = mask1 + cv2.inRange(hsv, lower_red, upper_red)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel)

        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key = lambda x: cv2.contourArea(x))
            cv2.drawContours(frame, [cnt], 0, (0, 0, 0), 5)             
            
            M = cv2.moments(cnt)
            center_x = int((M['m10']/M['m00']) * x_mult)
            center_y = int((M['m01']/M['m00']) * y_mult)

            pyautogui.moveTo(center_x, center_y)

            cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)

        cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()