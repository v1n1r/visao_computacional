import cv2

cap = cv2.VideoCapture(0)
lar = 420
alt = 360

while True:

    ret, video = cap.read()

    video1 = cv2.resize(video, (lar,alt), fx=0, fy=0, interpolation=cv2.INTER_AREA)

    cv2.imshow('Video', video1)

    key =  cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.relese()

# 16/01 https://www.youtube.com/watch?v=stCuBcDYatA - 16:39
