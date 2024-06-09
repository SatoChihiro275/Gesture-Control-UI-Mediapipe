import cv2
import mediapipe as mp
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#####関数の定義#####

#指先とポインターの描画
def pinch_pointer(f0_pos, f1_pos, pt_pos):
    cv2.circle(image, f0_pos, 5, color = (0, 0, 0), thickness=-1)
    cv2.circle(image, f1_pos, 5, color = (0, 0, 0), thickness=-1)
    cv2.circle(image, pt_pos, 5, color = (0, 0, 255), thickness=-1)
    cv2.circle(image, pt_pos, 16, color = (0, 0, 255), thickness=1)
    cv2.circle(image, pt_pos, 27, color = (0, 255, 0), thickness=1)

#keybord座標の取得
def KeyBord_location(x, y, deltaX, deltaY, S=10):
    key_x = -1
    key_y = -1
    for i in range(4):
        if (x > S+deltaX*i) and (x < S+deltaX*(i+1)): key_x = i
    for i in range(3):
        if (y > S+deltaY*(i+1.5)) and (y < S+deltaY*(i+2.5)) : key_y = i
    return key_x, key_y

# TenKeyの描画
def tenkey(X, Y, deltaX, deltaY, S=10):
    # 出力ボックスの描画
    cv2.rectangle(image, (S, S), (X-S, int(S+deltaY)), color=(255, 255, 255), thickness=-1)
    cv2.rectangle(image, (S, S), (X-S, int(S+deltaY)), color=(255, 0, 0), thickness=2)

    # KeyBoardの描画
    for i in range(2,6):
        cv2.line(image, (S, int(S+deltaY*(i-0.5))), (X-S, int(S+deltaY*(i-0.5))), (255, 255, 255), thickness=3)
    for i in range(5):
        cv2.line(image, (int(S+deltaX*i), int(S+deltaY*(1.5))), (int(S+deltaX*i), int(Y-S-deltaY*0.5)), (255, 255, 255), thickness=3)
    for i in range(2,6):
        cv2.line(image, (S, int(S+deltaY*(i-0.5))), (X-S, int(S+deltaY*(i-0.5))), (255, 0, 0), thickness=2)
    for i in range(5):
        cv2.line(image, (int(S+deltaX*i), int(S+deltaY*1.5)), (int(S+deltaX*i), int(Y-S-deltaY*0.5)), (255, 0, 0), thickness=2)
    #cv2.putText(image, "", (int(S+deltaX*0.35), int(S+deltaY*2.73)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    #cv2.putText(image, "", (int(S+deltaX*0.35), int(S+deltaY*2.73)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(image, "7", (int(S+deltaX*1.35), int(S+deltaY*2.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(image, "7", (int(S+deltaX*1.35), int(S+deltaY*2.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(image, "8", (int(S+deltaX*2.35), int(S+deltaY*2.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(image, "8", (int(S+deltaX*2.35), int(S+deltaY*2.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(image, "9", (int(S+deltaX*3.35), int(S+deltaY*2.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(image, "9", (int(S+deltaX*3.35), int(S+deltaY*2.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    cv2.putText(image, "BS", (int(S+deltaX*0.25), int(S+deltaY*3.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(image, "BS", (int(S+deltaX*0.25), int(S+deltaY*3.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(image, "4", (int(S+deltaX*1.35), int(S+deltaY*3.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(image, "4", (int(S+deltaX*1.35), int(S+deltaY*3.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(image, "5", (int(S+deltaX*2.35), int(S+deltaY*3.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(image, "5", (int(S+deltaX*2.35), int(S+deltaY*3.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(image, "6", (int(S+deltaX*3.35), int(S+deltaY*3.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(image, "6", (int(S+deltaX*3.35), int(S+deltaY*3.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    cv2.putText(image, "0", (int(S+deltaX*0.35), int(S+deltaY*4.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(image, "0", (int(S+deltaX*0.35), int(S+deltaY*4.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(image, "1", (int(S+deltaX*1.35), int(S+deltaY*4.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(image, "1", (int(S+deltaX*1.35), int(S+deltaY*4.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(image, "2", (int(S+deltaX*2.35), int(S+deltaY*4.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(image, "2", (int(S+deltaX*2.35), int(S+deltaY*4.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(image, "3", (int(S+deltaX*3.35), int(S+deltaY*4.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(image, "3", (int(S+deltaX*3.35), int(S+deltaY*4.23)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

###################
numbers = ""
pt_on = False
flag_off = True
f0_pos = (0, 0)
f1_pos = (0, 0)
pt_pos = (0, 0)

# For webcam input:
cap = cv2.VideoCapture(0)  # カメラのID指定
# Handsの設定
with mp_hands.Hands(
    model_complexity=0,           # モデルの複雑さ
    max_num_hands=1,              # 最大検出数
    min_detection_confidence=0.5, # 検出信頼度
    min_tracking_confidence=0.5   # 追跡信頼度
    ) as hands:
  while cap.isOpened():
    # フレーム画像の取得
    success, image = cap.read()
    Y, X = image.shape[:2] #画像サイズの取得
    S = 10
    deltaX = (X - 2*S) / 4
    deltaY = (Y - 2*S) / 5
    # imageの左右反転
    image = cv2.flip(image, 1)
    # 取得できなかった時の処理
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    # 書き込み不可の処理
    image.flags.writeable = False
    # BGRをRGBに変換
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 検出処理の実行
    results = hands.process(image)

    # Draw the hand annotations on the image.
    # 書き込み可の処理
    image.flags.writeable = True
    # RGBをBGRに変換
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      #検出した手の数分繰り返し
      for hand_landmarks in results.multi_hand_landmarks:
        #指の座標の取得
        f0 = hand_landmarks.landmark[4]
        f1 = hand_landmarks.landmark[8]
        f0_pos = (int(f0.x * X), int(f0.y * Y))
        f1_pos = (int(f1.x * X), int(f1.y * Y))
        pt_pos = (int((f0.x*X + f1.x*X) / 2), int((f0.y*Y + f1.y*Y) / 2))

        #keybord座標の取得
        key_x, key_y = KeyBord_location(pt_pos[0], pt_pos[1], deltaX, deltaY)

         # 判断(pinch)
        r = math.sqrt(pow(f0_pos[0]-f1_pos[0], 2) + pow(f0_pos[1] - f1_pos[1], 2)) /2
        if 16 > r: 
          pt_on = True
        else:
          pt_on = False
          if r > 27: flag_off = True

        # 入力
        if flag_off == True:
          if pt_on == True:
            if (key_x==0) and (key_y==2):
              numbers += str(0)
              flag_off = False
            elif (key_x==1) and (key_y==2):
              numbers += str(1)
              flag_off = False
            elif (key_x==2) and (key_y==2):
              numbers += str(2)
              flag_off = False
            elif (key_x==3) and (key_y==2):
              numbers += str(3)
              flag_off = False
            elif (key_x==1) and (key_y==1):
              numbers += str(4)
              flag_off = False
            elif (key_x==2) and (key_y==1):
              numbers += str(5)
              flag_off = False
            elif (key_x==3) and (key_y==1):
              numbers += str(6)
              flag_off = False
            elif (key_x==1) and (key_y==0):
              numbers += str(7)
              flag_off = False
            elif (key_x==2) and (key_y==0):
              numbers += str(8)
              flag_off = False
            elif (key_x==3) and (key_y==0):
              numbers += str(9)
              flag_off = False
            elif (key_x==0) and (key_y==1):
              numbers = numbers[:-1]
              flag_off = False        
    
    # KeyBoardの入力
    n = cv2.waitKey(1)
    if (n >= 48) and (n < 58):
      numbers += str(n-48)
    if n == 8:
      numbers = numbers[:-1]

    #指先とポインターの描画
    pinch_pointer(f0_pos, f1_pos, pt_pos)
    
    # TenKeyの描画
    tenkey(X, Y, deltaX, deltaY)

    #数字の表示
    cv2.putText(image, numbers, (int(S+deltaX*0.1), int(S+deltaY*0.73)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.0, color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
   
    # Full screen
    cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('MediaPipe Hands', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # imageの描画
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(1) & 0xFF == 27:
      break
cap.release()