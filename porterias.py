import cv2
import numpy as np

captura = cv2.VideoCapture(1)

while True:
    # Cámara
    ret, fotograma = captura.read()

    #fotograma = cv2.imread("campo.jpg")

    gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gris, 190, 255, cv2.THRESH_BINARY)
    canny = cv2.Canny(thresh, 50, 150)

    lineas =  cv2.HoughLinesP(canny, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    if lineas is not None:
        for line in lineas:
                x1, y1, x2, y2 = line[0]
                # Ángulo de la línea
                angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                # Dibujar la línea si es vertical (ángulo en el rango de 85 a 95 grados o -85 a -95 grados)
                if np.abs(angle) > 80 and np.abs(angle) < 100:
                    cv2.line(fotograma, (x1, y1), (x2, y2), (0, 0, 255), 3)


    cv2.imshow("Fotograma", fotograma)
    #cv2.imshow("Canny", canny)
    #cv2.imshow("Thresh", thresh)

    # Si se presiona la tecla 'q', salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

captura.release()
cv2.destroyAllWindows()