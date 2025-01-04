import cv2
import numpy as np

#imagem = np.zeros((400,400,3), dtype='uint8')

imagem = cv2.imread('pic.png')

#cv2.line(imagem, (0,170), (400,170), (0, 255, 0), 5)

#cv2.rectangle(imagem, (30,30), (200, 200), (0, 255,0), 5)

#cv2.circle(imagem, (200,200), 100, (255, 0, 0), 3)

font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(imagem, 'JosephDream', (50,50), font, 0.8,(0,0,0), 2, cv2.LINE_AA)

cv2.imshow('Imagem', imagem)

cv2.waitKey(0)
cv2.destroyAllWindows()