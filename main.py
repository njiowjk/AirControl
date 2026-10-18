import mediapipe as mp
import time
import cv2
import pyautogui
import math

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
screen_width, screen_height = pyautogui.size()
control_enabled = False
smooth_x = None
smooth_y = None
MIN_ALPHA = 0.08
MAX_ALPHA = 0.55
FAST_DISTANCE = 120

pyautogui.PAUSE = 0
click_state = "idle"
EXTEND_THRESHOLD = 0.75
PINCH_THRESHOLD = 0.3
RELEASE_THRESHOLD = 0.8

ACTIVE_X_MIN = 0.20
ACTIVE_X_MAX = 0.80

ACTIVE_Y_MIN = 0.30
ACTIVE_Y_MAX = 0.60

SCROLL_START_THRESHOLD = 0.06
SCROLL_STEP = 0.03

scroll_active = False
scroll_start_y = None
scroll_last_y = None
scroll_remainder = 0.0
SCROLL_LOST_FRAMES = 5
scroll_lost_frames = 0
SCROLL_INTERVAL = 0.01
last_scroll_time = 0.0

while True:
    ret, frame = cap.read()

    if not ret:
        print("画像を取得できませんでした")
        break

    # カメラ映像を左右反転
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )
    timestamp_ms = int(time.monotonic() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    


    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        thumb_tip = hand[4]
        middle_tip = hand[12]
        wrist = hand[0]
        middle_base = hand[9]
        height, width = frame.shape[:2]
        connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        dx = (thumb_tip.x - middle_tip.x) * width
        dy = (thumb_tip.y - middle_tip.y) * height
        distance = math.hypot(dx, dy)

        

        middle_length = math.hypot(
        (middle_tip.x - middle_base.x) * width,
        (middle_tip.y - middle_base.y) * height
    )
        palm_length = math.hypot(
        (middle_base.x - wrist.x) * width,
        (middle_base.y - wrist.y) * height
    )
        

        if palm_length > 0:
            middle_ratio = middle_length / palm_length
            pinch_ratio = distance / palm_length

            # 人差し指、薬指、小指の伸び具合
            def finger_ratio(tip, base):
                return math.hypot(
                    (hand[tip].x - hand[base].x) * width,
                    (hand[tip].y - hand[base].y) * height
                ) / palm_length

            index_ratio = finger_ratio(8, 5)
            ring_ratio = finger_ratio(16, 13)
            pinky_ratio = finger_ratio(20, 17)

            # 人差し指と中指だけ伸ばしているか
            scroll_pose = (
                index_ratio > 0.75
                and middle_ratio > EXTEND_THRESHOLD
                and ring_ratio < 0.65
                and pinky_ratio < 0.65
                and pinch_ratio > RELEASE_THRESHOLD
            )


            # 2本指の平均Y座標
            scroll_y = (hand[8].y + hand[12].y) / 2


            if control_enabled and click_state != "wait_release":

                if scroll_pose:
                    scroll_lost_frames = 0

                    if scroll_start_y is None:
                        scroll_start_y = scroll_y
                        scroll_last_y = scroll_y

                    elif not scroll_active:
                        if abs(scroll_y - scroll_start_y) > SCROLL_START_THRESHOLD:
                            scroll_active = True
                            scroll_last_y = scroll_y
                            click_state = "idle"
                            print("スクロール開始")

                    
                    else:
                        # スクロール開始位置からの移動量
                        offset_y = scroll_y - scroll_start_y

                        # 一定以上離れている間はスクロール
                        if abs(offset_y) > SCROLL_START_THRESHOLD:
                            now = time.monotonic()

                            if now - last_scroll_time >= SCROLL_INTERVAL:

                                if offset_y > 0:
                                    pyautogui.scroll(-1)  # 下
                                else:
                                    pyautogui.scroll(1)   # 上

                                last_scroll_time = now

                else:
                    scroll_lost_frames += 1
                    scroll_last_y = None

                    # 5フレーム連続で認識できなければ解除
                    if scroll_lost_frames >= SCROLL_LOST_FRAMES:
                        scroll_active = False
                        scroll_start_y = None
                        scroll_remainder = 0.0

            else:
                scroll_active = False
                scroll_start_y = None
                scroll_last_y = None
                scroll_remainder = 0.0
                scroll_lost_frames = 0
            

            if control_enabled and not scroll_active:
            

                # 通常状態 → クリック準備
                if click_state == "idle":
                    if middle_ratio > EXTEND_THRESHOLD and pinch_ratio > RELEASE_THRESHOLD:
                        click_state = "armed"
                        print("クリック準備")

                # クリック準備 → 左クリック
                elif click_state == "armed":
                    if pinch_ratio < PINCH_THRESHOLD:
                        pyautogui.click()
                        print("LEFT CLICK")
                        click_state = "wait_release"

                # 指を離すまで待機
                elif click_state == "wait_release":
                    if pinch_ratio > RELEASE_THRESHOLD or middle_ratio < 0.65:
                        click_state = "idle"
                        print("通常状態に戻りました")
            elif not control_enabled:
                click_state = "idle"


        else:
            click_state = "idle"
            scroll_active = False
            scroll_start_y = None
            scroll_last_y = None
            scroll_remainder = 0.0
            scroll_lost_frames = 0


        
        
        

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
        # 人差し指の先端を取得
        index_tip = hand[7]
        cv2.circle(
            frame,
            (int(index_tip.x * width), int(index_tip.y * height)),
            10,
            (0, 0, 255),
            -1
        )

        # 操作範囲を0〜1に変換
        normalized_x = (index_tip.x - ACTIVE_X_MIN) / (ACTIVE_X_MAX - ACTIVE_X_MIN)
        normalized_y = (index_tip.y - ACTIVE_Y_MIN) / (ACTIVE_Y_MAX - ACTIVE_Y_MIN)

        # 範囲外の座標を0〜1に制限
        normalized_x = max(0, min(1, normalized_x))
        normalized_y = max(0, min(1, normalized_y))

        # Macの画面座標に変換
        mouse_x = int(1 + normalized_x * (screen_width - 3))
        mouse_y = int(1 + normalized_y * (screen_height - 3))

        # 操作が有効なときだけ移動
        
        if control_enabled and click_state != "wait_release" and not scroll_active:
            if smooth_x is None:
                smooth_x = mouse_x
                smooth_y = mouse_y
            else:
                # 目標位置までの距離を計算
                cursor_distance = math.hypot(
                    mouse_x - smooth_x,
                    mouse_y - smooth_y
                )

                # 距離に応じてalphaを変更
                ratio = min(cursor_distance / FAST_DISTANCE, 1.0)
                current_alpha = MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * ratio

                # カーソルを移動
                smooth_x += current_alpha * (mouse_x - smooth_x)
                smooth_y += current_alpha * (mouse_y - smooth_y)

            pyautogui.moveTo(int(smooth_x), int(smooth_y))


        #if result.hand_landmarks:
            #print("手を検出しました！")
        if palm_length > 0:
            cv2.putText(
                frame,
                f"Pinch ratio: {pinch_ratio:.2f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )
    else:
        click_state = "idle"
    
    height, width = frame.shape[:2]

    cv2.rectangle(
        frame,
        (int(ACTIVE_X_MIN * width), int(ACTIVE_Y_MIN * height)),
        (int(ACTIVE_X_MAX * width), int(ACTIVE_Y_MAX * height)),
        (255, 255, 0),
        2
    )

    cv2.imshow("AirControl", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('m'):
        control_enabled = not control_enabled
        smooth_x = None
        smooth_y = None
        click_state = "idle"

        scroll_active = False
        scroll_start_y = None
        scroll_last_y = None
        scroll_remainder = 0.0

        print("カーソル操作:", "ON" if control_enabled else "OFF")

    

landmarker.close()
cap.release()
cv2.destroyAllWindows()
