import cv2

img = cv2.imread("pic.png", cv2.IMREAD_GRAYSCALE) # Abrir a imagem original
img1 = cv2.imread("pic.png", cv2.IMREAD_COLOR) # Abrir a imagem original

# Dados de redimensionamento para imagem a cor
escala_per = 60
largura = int(img.shape[1]*escala_per/100)
altura = int(img.shape[0]*escala_per/100)

dim = (largura,altura)

# Dados de redimensionamento para imagem cinza
escala_per = 60
largura = int(img.shape[1]*escala_per/100)
altura = int(img.shape[0]*escala_per/100)

dim = (largura,altura)

img2 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA) #imagem redimensioada

cv2.imshow("Imagem", img2)

img3= cv2.resize(img, dim, interpolation=cv2.INTER_AREA) #imagem redimensioada

cv2.imshow("ImagemCinza", img3)



cv2.waitKey(0)
cv2.destroyAllWindows()