import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from skimage.color import rgb2lab

class SegmentationProcessor(QWidget):
    def __init__(self,image_path):
        super().__init__()
        self.image_path = image_path
        self.segmented_image = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Tambahkan label untuk menampilkan hasil segmentasi
        self.segmentation_result_label = QLabel(self)
        layout.addWidget(self.segmentation_result_label)

        # Tambahkan tombol "Apply Segmentation"
        apply_segmentation_button = QPushButton("Apply Segmentation", self)
        apply_segmentation_button.clicked.connect(self.apply_segmentation)  # Hubungkan dengan metode apply_segmentation yang ada di kelas ini
        layout.addWidget(apply_segmentation_button)

        self.setLayout(layout)

    def apply_segmentation(self):
        if self.image_path:
            # Baca gambar dari path yang disimpan di self.image_path
            original_image = cv2.imread(self.image_path)
        
            # Konversi gambar ke format Lab
            lab_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2Lab)
        
            # Tentukan batasan warna untuk segmentasi (merah untuk buah matang, hijau untuk buah belum matang)
            lower_red = np.array([0, 0, 0])
            upper_red = np.array([120, 255, 255])
            lower_green = np.array([40, 40, 40])
            upper_green = np.array([90, 255, 255])

            # Lakukan segmentasi berdasarkan warna merah
            red_mask = cv2.inRange(lab_image, lower_red, upper_red)

            # Lakukan segmentasi berdasarkan warna hijau
            green_mask = cv2.inRange(lab_image, lower_green, upper_green)

            # Gabungkan masker merah dan hijau
            combined_mask = cv2.bitwise_or(red_mask, green_mask)

            # Aplikasikan masker ke gambar asli
            segmented_image = cv2.bitwise_and(original_image, original_image, mask=combined_mask)

            # Konversi gambar hasil segmentasi ke format RGB
            segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

            # Konversi gambar ke format QImage agar bisa ditampilkan di QLabel
            height, width, channel = segmented_image_rgb.shape
            bytes_per_line = 3 * width
            q_img = QImage(segmented_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.segmentation_result_label.setPixmap(pixmap)
        else:
            # Menampilkan pesan kesalahan jika gambar belum diunggah
            QMessageBox.warning(self, 'Peringatan', 'Silakan upload gambar terlebih dahulu!')
            # Atau, alternatifnya, Anda bisa mengosongkan QLabel untuk menghilangkan gambar hasil segmentasi
            # self.segmentation_result_label.clear()

    
    def get_segmented_image(self):
        return self.segmented_image
