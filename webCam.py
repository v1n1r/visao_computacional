import cv2

cap = cv2.VideoCapture(0)
lar = 420
alt = 320

while True:
    ret, video = cap.read()

   

    videoRed = cv2.resize(video, (lar,alt), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    cinza = cv2.cvtColor(videoRed,cv2.COLOR_BGR2GRAY)


    cv2.imshow("Video", videoRed)
    cv2.imshow("VideoCINZA", cinza)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('a'):
        break

cap.release()
cv2.destroyAllWindows()
