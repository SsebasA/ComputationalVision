import cv2
import numpy as np

capture = cv2.VideoCapture(r"C:\Users\sebas\OneDrive\Documentos\VisionCompu\VideosMetricas\VideosMetricas\video_sol.mp4")
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

Sx = int(width / 2)
Sy = int(height / 2)

alpha = 0.8 # Pixeles claros
beta = 0.9 # Pixeles oscuros

primer_circulo = 0
contador_si_circulo = 0
contador_no_circulo = 0

while True:

    ret, frame = capture.read()
    if not ret:
        print("Error")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    contrast = cv2.convertScaleAbs(gray, 250, 0.9)

    blur = cv2.GaussianBlur(contrast, (3, 3), 0)

    adjusted = np.where(blur >= 128, blur * alpha, blur * beta)
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT_ALT,
                        1.5, 50,
                        minRadius= 16, maxRadius= 280,
                        param1=0.9, param2=0.9)
    
    if circles is None and primer_circulo == 1:
        contador_no_circulo = contador_no_circulo + 1

    if circles is not None:
        # Redondear los parámetros del círculo detectado y convertirlos a enteros
        if primer_circulo == 0:
            print("ya se vio el balón por primera vez")
            primer_circulo = 1
        contador_si_circulo = contador_si_circulo + 1
        circle = np.round(circles[0, :]).astype("int")[0]

        # Extraer las coordenadas y el radio del círculo
        (x, y, r) = circle

        error_x = Sx - x
        error_y = Sy - y

        print("Error X: " + str(error_x)+ " Error Y: " + str(error_y))

        # Dibujar el círculo en el fotograma
        cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), 4)
        cv2.line(frame, (0, 240), (640, 240), (255, 0, 0), 1)
        cv2.line(frame, (320, 640), (320, 0), (255, 0, 0), 1)

    cv2.circle(frame, (Sx, Sy), 1, (255, 0, 0), 4)

    # Mostrar el fotograma resultante
    cv2.imshow("Sphere detection", frame)
    cv2.imshow("blur", blur)
    cv2.imshow("Adjusted", adjusted)

    # Si se presiona la tecla 'q', salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        total = contador_si_circulo + contador_no_circulo
        print(f"Se detectó el balón un {(contador_si_circulo * 100) / total:.4f}% del tiempo total")
        print(f"No se detectó el balón un {(contador_no_circulo * 100) / total:.4f}% del tiempo total")

        break

capture.release()
cv2.destroyAllWindows()