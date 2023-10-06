import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap

class EdgeProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)
        self.setLayout(self.layout)

    def apply_edge_detection(self, image_path):
        # Baca gambar dari path
        image = cv2.imread(image_path, 0)
        # Lakukan edge detection menggunakan Canny
        edges = cv2.Canny(image, 100, 200)
        # Konversi hasil edge detection ke format QImage agar bisa ditampilkan di QLabel
        height, width = edges.shape
        bytes_per_line = 1 * width
        q_img = QImage(edges.data, width, height, bytes_per_line, QImage.Format_Indexed8)
        pixmap = QPixmap.fromImage(q_img)
        # Tampilkan hasil edge detection pada QLabel
        self.image_label.setPixmap(pixmap)
