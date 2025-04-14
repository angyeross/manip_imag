import cv2
import numpy as np
import matplotlib.pyplot as plt


# Función para añadir marca de agua
def agregar_marca_agua(imagen, texto="Angye Rosado"):
    overlay = imagen.copy()
    output = imagen.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (255, 255, 255)
    shadow_color = (0, 0, 0)

    (text_width, text_height), _ = cv2.getTextSize(texto, font, font_scale, thickness)
    x = imagen.shape[1] - text_width - 10
    y = imagen.shape[0] - 10

    cv2.putText(overlay, texto, (x + 1, y + 1), font, font_scale, shadow_color, thickness)
    cv2.putText(overlay, texto, (x, y), font, font_scale, color, thickness)

    alpha = 0.7
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


# Función para crear histograma como imagen
def crear_imagen_histograma(gray_img):
    plt.figure(figsize=(3, 3))
    plt.hist(gray_img.ravel(), bins=256, range=[0, 256], color='gray')
    plt.title('Histograma de Intensidad')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')
    plt.tight_layout()  # Ajusta para que no se corten etiquetas

    plt.savefig('temp_hist.png')  # Guardar sin recortar los ejes
    plt.close()

    hist_img = cv2.imread('temp_hist.png')
    hist_img = cv2.resize(hist_img, (300, 300))
    return hist_img


# Cargar imagen original
img = cv2.imread('s.jpg')
if img is None:
    print("No se pudo cargar la imagen. Asegúrate de que el archivo s.jpg existe.")
    exit()

# Redimensionar
img = cv2.resize(img, (300, 300))
img = agregar_marca_agua(img)
cv2.imwrite('s_original.jpg', img)

# 1. Histograma (ecualización)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist_eq = cv2.equalizeHist(gray)
hist_eq_color = cv2.cvtColor(hist_eq, cv2.COLOR_GRAY2BGR)
hist_eq_color = agregar_marca_agua(hist_eq_color)
cv2.imwrite('s_histograma.jpg', hist_eq_color)

# 1b. Imagen del histograma real
hist_img = crear_imagen_histograma(gray)
hist_img = agregar_marca_agua(hist_img)
cv2.imwrite('s_histograma_grafico.jpg', hist_img)

# 2. Espejo
mirror = cv2.flip(img, 1)
mirror = agregar_marca_agua(mirror)
cv2.imwrite('s_espejo.jpg', mirror)

# 3. Rotaciones
rotate_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
rotate_90 = agregar_marca_agua(rotate_90)
cv2.imwrite('s_rotacion_90.jpg', rotate_90)

rotate_180 = cv2.rotate(img, cv2.ROTATE_180)
rotate_180 = agregar_marca_agua(rotate_180)
cv2.imwrite('s_rotacion_180.jpg', rotate_180)

rotate_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
rotate_270 = agregar_marca_agua(rotate_270)
cv2.imwrite('s_rotacion_270.jpg', rotate_270)

# 4. Negativo
negativo = cv2.bitwise_not(img)
negativo = agregar_marca_agua(negativo)
cv2.imwrite('s_negativo.jpg', negativo)

# 5. Contraste
alpha = 1.5
beta = 0
contraste = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
contraste = agregar_marca_agua(contraste)
cv2.imwrite('s_contraste.jpg', contraste)

# 6. Blur
blur = cv2.GaussianBlur(img, (15, 15), 0)
blur = agregar_marca_agua(blur)
cv2.imwrite('s_blur.jpg', blur)

# Crear collage (3x3 + histograma)
row1 = np.hstack([img, hist_eq_color, mirror])
row2 = np.hstack([rotate_90, rotate_180, rotate_270])
row3 = np.hstack([negativo, contraste, blur])
row4 = np.hstack([hist_img, np.zeros_like(hist_img), np.zeros_like(hist_img)])

collage = np.vstack([row1, row2, row3, row4])
collage = agregar_marca_agua(collage)
cv2.imwrite('s_collage.jpg', collage)

# Mostrar collage
cv2.imshow('Collage de Modificaciones - Angye Rosado', collage)
cv2.waitKey(0)
cv2.destroyAllWindows()
