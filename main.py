import mediapipe as mp
import time
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1
)

landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

while True:
    ret, frame = cap.read()

    if not ret:
        print("画像を取得できませんでした")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )
    timestamp_ms = int(time.monotonic() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    


    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        height, width = frame.shape[:2]
        connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
       

        # 関節同士を線でつなぐ
        for connection in connections:
            point1 = hand[connection.start]
            point2 = hand[connection.end]

            x1 = int(point1.x * width)
            y1 = int(point1.y * height)

            x2 = int(point2.x * width)
            y2 = int(point2.y * height)

            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 21個の関節に点を描画
        for landmark in hand:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)


        #if result.hand_landmarks:
            #print("手を検出しました！")

    cv2.imshow("AirControl", frame)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()
