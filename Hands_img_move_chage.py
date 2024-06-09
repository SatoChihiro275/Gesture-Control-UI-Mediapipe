import cv2
import mediapipe as mp
import numpy as np
import math
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

################################################################

# 行列の掛け算
def composite(M1, M2):
    M1_tmp = np.array([M1[0], M1[1], [0,0,1]])
    M2_tmp = np.array([M2[0], M2[1], [0,0,1]])
    M_tmp = np.dot(M1_tmp, M2_tmp)
    M = np.array([M_tmp[0], M_tmp[1]])
    return M

# 変換行列の初期化
def M_reset(w, h, wb, hb):
    # 縦150pxとのスケール比
    s0 = 150 / h
    # スケール変換
    M1 = np.array([[s0, 0, 0], [0, s0, 0]], dtype=float)
    # 平行移動
    M2 = np.array([[1, 0, wb/2 - 75*w/h], [0, 1, 10]], dtype=float)
    #変換行列の合成
    M = composite(M2, M1)
    return M

# 変換行列の更新
def M_update(M, x, y, delta_x, delta_y, delta_s=1.0, delt_theta=0):
    # 平行移動(原点へ)
    M1 = np.array([[1, 0, -x],
                   [0, 1, -y]], dtype=float)
    # スケール変更・回転移動
    M2 = cv2.getRotationMatrix2D((0, 0), scale=delta_s, angle=delt_theta)

    # 平行移動(更新した座標へ)
    M3 = np.array([[1, 0, x + delta_x],
                   [0, 1, y + delta_y]], dtype=float)
    # 変換行列の合成
    M = composite(M1, M)
    M = composite(M2, M)
    M = composite(M3, M)
    return M

# キーパッドの描画
def keypad_imgchage(wb, hb):
    cv2.rectangle(image, (10, 10), (160, 160), color=(255, 255, 255), thickness=5)
    cv2.rectangle(image, (10, 10), (160, 160), color=(0, 0, 0), thickness=3)
    cv2.putText(image, "<", (15, 135), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=6.0, color=(255, 255, 255), thickness=5, lineType=cv2.LINE_AA)
    cv2.putText(image, "<", (15, 135), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=6.0, color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    cv2.rectangle(image, (wb-160, 10), (wb-10, 160), color=(255, 255, 255), thickness=5)
    cv2.rectangle(image, (wb-160, 10), (wb-10, 160), color=(0, 0, 0), thickness=3)
    cv2.putText(image, ">", (wb-145, 135), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=6.0, color=(255, 255, 255), thickness=5, lineType=cv2.LINE_AA)
    cv2.putText(image, ">", (wb-145, 135), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=6.0, color=(0, 0, 0), thickness=3, lineType=cv2.LINE_AA)

###################################################################
# 画像の読み込み
num_img = 0
path = "./image"
list_img = os.listdir(path)
len_img = len(list_img)
h = 150
w = 200

# 変換行列
M = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

# scissors用
flag_reset = True

# rock用
pt2_pos = (-100, -100)
flag_rock = False
a = np.array([1, 0])

# pinch用
flag_pinch1_on = True
flag_pinch1_off = True
flag_pinch1 = False
flag_pinch2 =False
flag_pinch = False
pt_pos1 = (-50, -50)
pt_pos2 = (-50, -50)
pt_pos = (-50, -50)
###################################################################


# For webcam input:
cap = cv2.VideoCapture(0)  # カメラのID指定
# Handsの設定
with mp_hands.Hands(
    model_complexity=0,           # モデルの複雑さ
    max_num_hands=2,              # 最大検出数
    min_detection_confidence=0.5, # 検出信頼度
    min_tracking_confidence=0.5   # 追跡信頼度
    ) as hands:
  while cap.isOpened():
    # フレーム画像の取得
    success, image = cap.read()
    # 取得できなかった時の処理
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    #imageのshape取得
    hb, wb = image.shape[:2]
    # Flip the image horizontally for a selfie-view display.(imageの左右反転)
    image = cv2.flip(image, 1)

    # To improve performance, optionally mark the image as not writeable to　pass by reference.(書き込み不可処理)
    image.flags.writeable = False
    # BGRをRGBに変換
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 検出処理の実行
    results = hands.process(image)

    # Draw the hand annotations on the image.(書き込み可の処理)
    image.flags.writeable = True
    # RGBをBGRに変換
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    # imgの読み込み
    if flag_pinch1_on == True:
        if num_img < 0:
            num_img += len(list_img)
        elif num_img >= len(list_img):
            num_img += -len(list_img)
        name_img = list_img[num_img]
        img = cv2.imread(path + "/" + name_img)
        h, w = img.shape[:2]
        flag_pinch1_on = False
        flag_reset = True

    # 初期化
    if flag_reset==True:
        M = M_reset(w, h, wb, hb)
        B = np.array([[w/2], [h/2], [1]], dtype=float)
        A = np.dot(M, B)
        # imageの位置
        x0_rock = A[0][0]
        y0_rock = A[1][0]
        x0_pinch = A[0][0]
        y0_pinch = A[1][0]
        x = A[0][0]
        y = A[1][0]
        delta_x = 0
        delta_y = 0
        delta_s = 1.0
        delta_theta = 0
        r_img = int(A[1][0]) -10
        r0 = math.nan
        a0 = np.array([math.nan, math.nan])

    flag_pinch = False
    roop = 0
    # 検出アリの場合:
    if results.multi_hand_landmarks:
      # 手の検出数分の繰り返し処理:
      for hand_landmarks in results.multi_hand_landmarks:
        roop += 1
        
        #landmarksの取得
        lm0  = hand_landmarks.landmark[0]
        lm4  = hand_landmarks.landmark[4]
        lm5  = hand_landmarks.landmark[5]
        lm8  = hand_landmarks.landmark[8]
        lm9  = hand_landmarks.landmark[9]
        lm12 = hand_landmarks.landmark[12]
        lm13 = hand_landmarks.landmark[13]
        lm16 = hand_landmarks.landmark[16]
        lm17 = hand_landmarks.landmark[17]
        lm20 = hand_landmarks.landmark[20]

        #landmarksの座標の取得
        lm4_pos = (int(lm4.x * wb), int(lm4.y * hb))
        lm8_pos = (int(lm8.x * wb), int(lm8.y * hb))
        lm12_pos = (int(lm12.x * wb), int(lm12.y * hb))
        lm16_pos = (int(lm16.x * wb), int(lm16.y * hb))
        lm20_pos = (int(lm20.x * wb), int(lm20.y * hb))

        # 基準点との距離(pinch)
        r_pinch = math.sqrt(pow(lm8_pos[0] - lm4_pos[0], 2) + pow(lm8_pos[1] - lm4_pos[1], 2)) / 2

        # roop==1について:
        if roop==1:
            # 判断の基準点(rock)
            pt1_pos = (int((lm0.x*wb + lm5.x*wb*2) / 3), int((lm0.y*hb + lm5.y*hb*2) / 3))
            pt2_pos = (int((lm0.x*wb + lm9.x*wb*2) / 3), int((lm0.y*hb + lm9.y*hb*2) / 3))
            pt3_pos = (int((lm0.x*wb + lm13.x*wb*2) / 3), int((lm0.y*hb + lm13.y*hb*2) / 3))
            pt4_pos = (int((lm0.x*wb + lm17.x*wb*2) / 3), int((lm0.y*hb + lm17.y*hb*2) / 3))
            
            # 基準点の距離(rock)
            r1 = math.sqrt(pow(pt1_pos[0]-lm8_pos[0], 2) + pow(pt1_pos[1] - lm8_pos[1], 2)) / 2
            r2 = math.sqrt(pow(pt2_pos[0]-lm12_pos[0], 2) + pow(pt2_pos[1] - lm12_pos[1], 2)) /2
            r3 = math.sqrt(pow(pt3_pos[0]-lm16_pos[0], 2) + pow(pt3_pos[1] - lm16_pos[1], 2)) /2
            r4 = math.sqrt(pow(pt4_pos[0]-lm20_pos[0], 2) + pow(pt4_pos[1] - lm20_pos[1], 2)) /2

            # pt2とimgの距離(rock)
            r_img2rock = math.sqrt(pow(pt2_pos[0] - x, 2) + pow(pt2_pos[1] - y, 2))

            # ポインターの位置(pinch1)
            pt_pos1 = (int((lm4_pos[0]+lm8_pos[0]) / 2), int((lm4_pos[1]+lm8_pos[1]) / 2))

            # 判断(rock)
            flag_rock = False
            if (r1<20)and(r2<20)and(r3<20)and(r4<20):
                flag_rock = True
                if (r_img2rock < r_img):
                    # 値の更新(rock)
                    B = np.array([[w/2],[h/2],[1]], dtype=float)
                    A = np.dot(M, B)
                    delta_x = pt2_pos[0] - x0_rock
                    delta_y = pt2_pos[1] - y0_rock
                    x = A[0][0] + delta_x
                    y = A[1][0] + delta_y
                    if (delta_x>50)or(delta_y>50):
                        delta_x = 0
                        delta_y = 0
                    # 更新
                    M = M_update(M, x, y, delta_x, delta_y, 1.0, 0)
            
            # 一時保存(rock)
            x0_rock = pt2_pos[0]
            y0_rock = pt2_pos[1]

            # 判断(scissors)
            if r_pinch > 40:
                if (r1>40)and(r2>40)and(r3<20)and(r4<20):
                    flag_reset = True
            else:
                flag_reset = False

            # 判断(pinch1+change)
            if r_pinch < 15:
                flag_pinch1 = True
            else:
                flag_pinch1 = False
                if r_pinch > 30:
                    flag_pinch1_off = True

            # 判断(change)
            if (flag_pinch1==True)and(flag_pinch1_off==True):
                if 10 < pt_pos1[1] < 160:
                    if 10 < pt_pos1[0] < 160:
                        num_img += -1
                        flag_pinch1_on = True
                        flag_pinch1_off = False
                    elif (wb-160) < pt_pos1[0] < (wb-10):
                        num_img += 1
                        flag_pinch1_on = True
                        flag_pinch1_off = False
                        #print("change")
        
        # roop==2について:
        if roop==2:
            # ポインターの位置(pinch2)
            pt_pos2 = (int((lm4_pos[0]+lm8_pos[0]) / 2), int((lm4_pos[1]+lm8_pos[1]) / 2))

            # 判断(pinch2)
            if r_pinch < 15:
                flag_pinch2 = True
            else:
                flag_pinch2 = False

            # ポインターの位置(pinch)
            pt_pos = (int((pt_pos1[0]+pt_pos2[0]) / 2), int((pt_pos1[1]+pt_pos2[1]) / 2))

            # ptとimgの距離(pinch)
            r_img2pinch = math.sqrt(pow(pt_pos[0] - x, 2) + pow(pt_pos[1] - y, 2))

            # pt_pos1とpt_pos2の距離(pinch)
            r_pinch2pinch = math.sqrt(pow(pt_pos1[0] - pt_pos2[0], 2) + pow(pt_pos1[1] - pt_pos2[1], 2))

            # 判断(pinch)
            if (flag_pinch1==True)and(flag_pinch2==True):
                flag_pinch = True
                if (r_img2pinch < r_img):
                    # 値の更新(pinch)
                    B = np.array([[w/2],[h/2],[1]], dtype=float)
                    A = np.dot(M, B)
                    delta_x = pt_pos[0] - x0_pinch
                    delta_y = pt_pos[1] - y0_pinch
                    x = pt_pos[0]
                    y = pt_pos[1]
                    if (delta_x>50)or(delta_y>50):
                        delta_x = 0
                        delta_y = 0
                    
                    delta_s = 1.0
                    delta_theta = 0
                    if np.isnan(r0)==False:
                            delta_s = r_pinch2pinch / r0
                    if (delta_s<0.8)or(delta_s>1.2):
                        delta_s = 1.0                    
                    a = np.array([pt_pos1[0]-pt_pos2[0], pt_pos1[1]-pt_pos2[1]])
                    delta_theta = math.degrees(np.arcsin((a[0] * a0[1] - a[1] * a0[0]) / (np.linalg.norm(a) * np.linalg.norm(a0))))
                    if (delta_theta<-15)or(delta_theta>15):
                        delta_theta = 0
                    
                    # 更新
                    M = M_update(M, x, y, delta_x, delta_y, delta_s, delta_theta)
                    x = A[0][0] + delta_x
                    y = A[1][0] + delta_y
                    # 画像半径の導出
                    B = np.array([[w/2],[0],[1]], dtype=float)
                    A = np.dot(M, B)
                    r_img = np.sqrt(pow(x - A[0][0], 2) + pow(y - A[1][0], 2))
                    
            # 一時保存(pinch)
            x0_pinch = pt_pos[0]
            y0_pinch = pt_pos[1]
            r0 = r_pinch2pinch
            a0 = a 
    
    #print("delta_x:", delta_x, "  delta_y:", delta_y, "  delta_s:", delta_s, "  delta_theta:", delta_theta)
    
    # imgをimageに挿入
    image = cv2.warpAffine(img, M, dsize=(wb, hb), dst=image, borderMode=cv2.BORDER_TRANSPARENT, flags=cv2.INTER_NEAREST)
    
    #画像領域の描画
    cv2.circle(image, (int(x), int(y)), 5, color = (255, 0, 0), thickness=-1)
    if np.isnan(r_img)==False:
        cv2.circle(image, (int(x), int(y)), int(r_img), color = (255, 0, 0), thickness=1)

    # キーパッドの描画
    keypad_imgchage(wb, hb)

    # ポインターの描画(rock)
    cv2.circle(image, pt2_pos, 5, color = (0, 0, 255), thickness=-1)

    #rockの信号
    if flag_rock==True:
        cv2.circle(image, pt2_pos, 75, color = (0, 255, 0), thickness=5)

    #ポインターの描画(pinch1)
    cv2.circle(image, pt_pos1, 5, color = (0, 0, 0), thickness=-1)

    #ポインターの描画(pinch2)
    if roop==2:
        cv2.circle(image, pt_pos2, 5, color = (0, 0, 0), thickness=-1)

    #ポインターの描画(pinch)
    if roop==2:
        cv2.circle(image, pt_pos, 5, color = (0, 0, 255), thickness=-1)
    if flag_pinch==True:
        cv2.circle(image, pt_pos1, 20, color = (0, 255, 0), thickness=5)
        cv2.circle(image, pt_pos2, 20, color = (0, 255, 0), thickness=5)

    # scissorsの信号
    if flag_reset == True:
        cv2.circle(image, pt2_pos, 75, color = (255, 0, 0), thickness=5)

    # changeの信号
    if flag_pinch1_on == True:
        cv2.circle(image, pt_pos1, 25, color = (255, 0, 0), thickness=5)
    
    # Full screen
    cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('MediaPipe Hands', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # imageの描画
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(1) & 0xFF == 27:
      break
cap.release()