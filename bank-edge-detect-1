#-*- coding:utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as image
from PIL import Image, ImageDraw, ImageFont
import time
import numpy as np
from numpy.core.numeric import True_
import pandas as pd
import winsound as ws
from win32api import GetSystemMetrics
import os
import winsound
import pygame
from keras.models import load_model
from PIL import Image, ImageOps

def beep_sound():
    fre = 450
    dura = 10000
    winsound.Beep(fre, dura)

# 이미지에서 마우스 클릭 좌표 얻기
def MouseLeftClick(event, x, y, flags, param):
    if len(clicked_points) < 2:
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append((y, x))
            #print(y, x)

# 알람 및 판정
def Alarm(cre_int, vv):
    if vv > cre_int * 0.6:
        text_nor = '정상 : ' + str(vv) + ' %'
        draw.text((round(win_width/4)-90, 100), text_nor, font=font, fill=(255, 255, 0))

    elif vv <= cre_int * 0.6 and vv >= cre_int * 0.4:
        text_war = '경고 : ' + str(vv) + ' %'
        draw.text((round(win_width/4)-90, 100), text_war, font=font, fill=(0 ,255, 255))
        #beepsound()

    elif vv < cre_int * 0.4:
        text_unnor = '비정상 : ' + str(vv) + ' %'
        draw.text((round(win_width/4)-90, 100), text_unnor, font=font, fill=(0 ,0, 255))


# 선언 및 정의 : definition
# 마우스 클릭수 count 
clicked_points = []

# Canny Edge level
k1 = 100 # 100
k2 = 150 # 150

# bank 두께 수집
cre_value = []
ss = []
nr = 600

# 한글 폰트
fontpath = 'C:/Windows/Fonts/H2HDRM.TTF'
font = ImageFont.truetype(fontpath, 45)

# position 좌표
x = []
y = []
ccx = []
ccy = []
cre_interval = []

# 모니터 해상도 얻기
win_width = GetSystemMetrics(0)
win_height = GetSystemMetrics(1)

path = 'D:/kp-tech/PVC/bank/2021-10-04-13-05-53.mp4'
file_name = path[20:-4]

cap = cv2.VideoCapture(path)

#cap = cv2.VideoCaptur(0)
# 동영상 프레임 설정값 얻기
fps = cap.get(cv2.CAP_PROP_FPS)
cou = 0
while(cap.isOpened()):
    # 동영상 읽기
    ret, frame = cap.read()
    if ret:
        img = cv2.resize(frame, dsize=(round(win_width/2), round(win_height/2)-30))
        #img = cv2.rotate(img, cv2.ROTATE_180)

        cv2.imshow('Bank Image', img)

        # 마우스 2번 클릭으로 기준 영역 선택
        cv2.namedWindow('Bank Image')
        cv2.setMouseCallback('Bank Image', MouseLeftClick)

        if len(clicked_points) == 2:

            # 관심 영역 마우스 클릭 좌표 얻기
            x0 = clicked_points[0][0]
            y0 = clicked_points[0][1]

            x1 = clicked_points[1][0]
            y1 = clicked_points[1][1]

            # 관심 영역 이미지 얻기
            img_cre = img[x0:x1, y0:y1]
            #img_cre = cv2.resize(img_cre, dsize=(640, 480))
            img_cre = cv2.resize(img_cre, dsize=(round(win_width/2), round(win_height/2)-30))

            # line 넣을 RGB 이미지 생성
            img_line = cv2.resize(img, dsize=(round(win_width/2), round(win_height/2)-30))
            
            img_line[x0:x1, y0:y0+1, 1] = 255 # 좌측
            img_line[x0:x1, y0+round((y1-y0)/2):y0+round((y1-y0)/2)+1, 1] = 255 # 중앙
            img_line[x0:x1, y1:y1+1, 1] = 255 # 우측
            
            # Edge 이미지 생성
            img_gray = cv2.cvtColor(img_cre, cv2.COLOR_BGR2GRAY)

            img_blurred = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=0)

            #--③ OpenCV API를 이용한 정규화
            img_norm = cv2.normalize(img_blurred, None, 0, 255, cv2.NORM_MINMAX)
            #img_norm = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_HAMMING2)
            img_norm_bgr = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

            ret, img_thre = cv2.threshold(img_norm, 60, 85, cv2.THRESH_BINARY)
           
            img_canny = cv2.Canny(img_thre, k1, k2)

            # Edge 영역 좁게하기 위해 반사 영역 0처리
            img_canny[0:30, :] = 0 # 상부
            img_canny[-30:, :] = 0 # 하부

            # Edge 이미지에 상부/좌우 경계면 추가
            img_canny[10:-10, 10] = 255 # 좌측 경계
            #img_canny[0:20, :] = 0 # 상부 경계면 공백
            img_canny[10:-10,-10] = 255 # 우측 경계면 공백
            #img_canny[-20:, :] = 0 # 하부 경계면 공백

            contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #contours, _ = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            #print(contours[164])

            #perimeter = cv2.arcLength(contours[164], True)
            #print(perimeter)

            largestCnt = []
            for cnt in contours:
                if len(cnt) > len(largestCnt):
                    largestCnt = cnt

            M = cv2.moments(largestCnt)

            # 한글 표시
            img_line = Image.fromarray(img_line)
            draw = ImageDraw.Draw(img_line)

            try:
                xx = int(M["m10"] / M["m00"])
                yy = int(M["m01"] / M["m00"])

                x.append(xx)
                y.append(yy)

                if len(x) > 30:
                    cx = int(np.mean(x[-25:-1]))
                    cy = int(np.mean(y[-25:-1]))

                    ccx.append(cx)
                    ccy.append(cy)

                    # edge선 녹색으로 변경
                    img_contours = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
                    img_contours = cv2.drawContours(img_contours, contours, -2, (0, 255, 0), 2)

                    # 중심점 좌/우 가상 점
                    poi_right = int(1.5*cx)
                    poi_left = int(0.5*cx)

                    # 중심점, 좌점, 우점 화살표 표시
                    img_contours[cy-3:cy+3, poi_right-10:poi_right+10, 2] = 255
                    img_contours[cy-10:cy+10, poi_right-3:poi_right+3, 2] = 255

                    img_contours[cy-3:cy+3, cx-10:cx+10, 2] = 255
                    img_contours[cy-10:cy+10, cx-3:cx+3, 2] = 255

                    img_contours[cy-3:cy+3, poi_left-10:poi_left+10, 2] = 255
                    img_contours[cy-10:cy+10, poi_left-3:poi_left+3, 2] = 255

                    # bank 폭 비율 계산
                    if len(cre_interval) < 30:
                        cre_interval.append(int(abs(poi_left - poi_right)*100/310))
                        cre_int = int(min(cre_interval))
                        #print(cre_int)
                    else:
                        #cre_interval = cre_int
                        vv = int(abs(poi_left - poi_right)*100/310)
                        #print(cre_int, '   ', vv)
                        # bank 좌/우 가상점 위치 변화 및 판정 (알람)
                        Alarm(cre_int, vv)

                # 판정 결과 이미지 붙이기
                img_hor_0 = np.hstack((img, img_line))
                #cv2.imshow('Bank Between Rolls By S.H Kim', img_hor_0)

                img_hor_1 = np.hstack((img_norm_bgr, img_contours))
                
                #cv2.imshow('Position oF Bank Area By S.H Kim', img_hor_1)
                img_tot = np.vstack((img_hor_0, img_hor_1))
                cv2.imshow('Position oF Bank Area By S.H Kim', img_tot)

                # 입력 이미지 시간(분당)별 저장
                #img_res_gray = cv2.resize(img_norm_bgr, dsize=(150, 150))
                img_res = cv2.resize(img, dsize=(224, 224))

                if int(time.time())%3 == 0: # 5초 간격 저장
                    img_file_1 = './saved_img/' + time.strftime('%Y-%m-%d-%H-%M-%S')+'.jpg'
                    #cou += 1
                    #img_file_1 = 'D:/kp-tech/PVC/saved_img/bank-' + file_name +'-'+str(cou)+'.jpg'
                    if os.path.isfile(img_file_1) is False:
                        #cv2.imwrite(img_file_0, img)
                        cv2.imwrite(img_file_1, img_res)
                        #print(img_file_1)

            except:
                continue
        
    if cv2.waitKey(round(1000/fps)) & 0xFF == 27: #ord('q'):
        print(' 운전 중 중단 ... ... ')
        break

cap.release()
cv2.destroyAllWindows()

print('Having Finished ... ... ')
