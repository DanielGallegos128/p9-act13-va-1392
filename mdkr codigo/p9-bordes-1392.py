import cv2
print("Daniel Gallegos NC 1392 Problema 2")
# Cargar imagen
imagen = cv2.imread("mdkr imagenes/koala.jpeg")

# Comprobar imagen
if imagen is None:
    print("Error: no se pudo cargar la imagen.")
    exit()

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Convertir a imagen binaria mediante umbral
_, binaria = cv2.threshold(
    gris,
    127,
    255,
    cv2.THRESH_BINARY
)

# Detectar contornos
contornos, jerarquia = cv2.findContours(
    binaria,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Dibujar los contornos
resultado = imagen.copy()

cv2.drawContours(
    resultado,
    contornos,
    -1,
    (0, 255, 0),
    2
)

# Mostrar resultados
cv2.imshow("Koala original 1392", imagen)
cv2.imshow("Koala binario 1392", binaria)
cv2.imshow("Koala contornos detectados 1392", resultado)

# Guardar resultado
cv2.imwrite(
    "mdkr resultados/koala_contornos_1392.jpeg",
    resultado
)

print("Cantidad de contornos encontrados:", len(contornos))
print("Resultado guardado en mdkr resultados/koala_contornos_1392.jpeg")

# Esperar
cv2.waitKey(0)

# Cerrar ventanas
cv2.destroyAllWindows()
print("Daniel Gallegos NC 1392 Problema 2")