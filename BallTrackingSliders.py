import cv2
import numpy as np

def nothing(x):
    pass

# Configuración de la captura de video
capture = cv2.VideoCapture(r"C:\Users\sebas\OneDrive\Documentos\VisionCompu\VideosMetricas\VideosMetricas\video_sol.mp4")
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Configuración de los sliders
cv2.namedWindow('Controls')
cv2.createTrackbar('Alpha', 'Controls', 18, 30, nothing)  # Rango de 1 a 3 multiplicado por 10
cv2.createTrackbar('Beta', 'Controls', 7, 10, nothing)    # Rango de 0 a 1 multiplicado por 10
cv2.createTrackbar('DP', 'Controls', 10, 20, nothing)     # Rango de 1 a 2 multiplicado por 0.1
cv2.createTrackbar('MinDist', 'Controls', 50, 200, nothing) # Distancia mínima en píxeles
cv2.createTrackbar('MinRadius', 'Controls', 10, 300, nothing) # Radio mínimo en píxeles
cv2.createTrackbar('MaxRadius', 'Controls', 100, 300, nothing) # Radio máximo en píxeles
cv2.createTrackbar('Param1', 'Controls', 300, 500, nothing) # Umbral de Canny
cv2.createTrackbar('Param2', 'Controls', 9, 10, nothing)   # Umbral del acumulador (ajustado al rango de 0 a 1.0)

width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

Sx = int(width / 2)
Sy = int(height / 2)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Obtener valores de los sliders
    alpha = cv2.getTrackbarPos('Alpha', 'Controls') / 10.0
    beta = cv2.getTrackbarPos('Beta', 'Controls') / 10.0
    dp = cv2.getTrackbarPos('DP', 'Controls') / 10.0
    minDist = cv2.getTrackbarPos('MinDist', 'Controls')
    minRadius = cv2.getTrackbarPos('MinRadius', 'Controls')
    maxRadius = cv2.getTrackbarPos('MaxRadius', 'Controls')
    param1 = cv2.getTrackbarPos('Param1', 'Controls')
    param2 = cv2.getTrackbarPos('Param2', 'Controls') / 10.0

    # Asegurar que dp sea al menos 0.1
    dp = max(dp, 0.1)
    # Asegurar que minDist sea al menos 1 para evitar errores
    minDist = max(minDist, 1)
    # Asegurar que minRadius y maxRadius estén en el rango correcto
    minRadius = max(minRadius, 1)
    maxRadius = max(maxRadius, minRadius)
    # Asegurar que param1 esté en el rango adecuado
    param1 = min(max(param1, 0.1), 0.9)
    # Asegurar que param2 esté en el rango adecuado
    param2 = min(max(param2, 0.1), 0.9)

    contrast = cv2.convertScaleAbs(gray, alpha=1.0, beta=0)
    blur = cv2.GaussianBlur(contrast, (3, 3), 0)

    adjusted = np.where(blur >= 128, blur * alpha, blur * beta)
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

    circles = cv2.HoughCircles(adjusted, cv2.HOUGH_GRADIENT_ALT,
                              dp, minDist,
                              minRadius=minRadius, maxRadius=maxRadius,
                              param1=param1, param2=param2)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            error_x = Sx - x
            error_y = Sy - y

            print("Error X: " + str(error_x) + " Error Y: " + str(error_y))

            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), 4)

    cv2.circle(frame, (Sx, Sy), 1, (255, 0, 0), 4)

    # Mostrar las imágenes
    cv2.imshow("Sphere detection", frame)
    cv2.imshow("blur", blur)
    cv2.imshow("Adjusted", adjusted)

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Alpha: " + str(alpha))
        print("Beta: " + str(beta))

        print("dp: " + str(dp))
        print("minDist: " + str(minDist))
        print("minRadius: " + str(minRadius))
        print("maxRadius: " + str(maxRadius))
        print("param1: " + str(param1))
        print("param2: " + str(param2))
        break

capture.release()
cv2.destroyAllWindows()
