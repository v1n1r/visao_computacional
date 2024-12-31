import cv2

img = cv2.imread("pic.png", cv2.IMREAD_COLOR)

cv2.imshow("Imagem", img)

cv2.waitKey(0)
cv2.destroyAllWindows()