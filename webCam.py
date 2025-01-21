import cv2

cap = cv2.VideoCapture(0)
lar = 420
alt = 360

fourcc = cv2.VideoWriter_fourcc(*'XVID')
saida = cv2.VideoWriter('video_test.avi', fourcc, 20.0, (lar,alt))

while True:

    ret, video = cap.read()

    #video1 = cv2.resize(video, (lar,alt), fx=0, fy=0, interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(video, cv2.COLOR_BGR2HSV)

    saida.write(hsv)

    cv2.imshow('Video', video)

    key =  cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


saida.relese()
cap.relese()
cv2.destroyAllWindows()

# 16/01 https://www.youtube.com/watch?v=stCuBcDYatA - 16:39

