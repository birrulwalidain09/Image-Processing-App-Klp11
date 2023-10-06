import cv2
import numpy as np

def apply_vector_effect(image_path, output_path):
    # Baca gambar menggunakan OpenCV
    image = cv2.imread(image_path)

    # Periksa apakah gambar berhasil dibaca
    if image is None:
        print("Error: Gambar tidak dapat dibaca.")
        return

    # Ubah gambar ke citra abu-abu
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Deteksi tepi menggunakan metode Canny
    edges = cv2.Canny(gray_image, 50, 150)

    # Temukan garis menggunakan transformasi Hough
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

    # Gambar garis-garis pada gambar asli
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Simpan gambar hasil efek vektor
    cv2.imwrite(output_path, image)
