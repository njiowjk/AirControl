
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()


while True:
    ret, frame = cap.read()

    if not ret:
        print("画像を取得できませんでした")
        break

    cv2.imshow("AirControl", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
