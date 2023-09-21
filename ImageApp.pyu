import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class HistogramEqualizationApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Aplikasi Pengolahan Citra_Kelompok 11')
        self.setGeometry(100, 100, 800, 600)

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)

        self.btn_equalize = QPushButton('Histogram Equalization', self)
        self.layout.addWidget(self.btn_equalize)
        self.btn_equalize.clicked.connect(self.showFileSelectionPage)
        self.btn_equalize.setFixedWidth(self.btn_equalize.fontMetrics().boundingRect(self.btn_equalize.text()).width() + 20)

        self.btn_edge_detection = QPushButton('Edge Detection', self)
        self.layout.addWidget(self.btn_edge_detection)
        self.btn_edge_detection.clicked.connect(self.performEdgeDetection)
        self.btn_edge_detection.setFixedWidth(self.btn_edge_detection.fontMetrics().boundingRect(self.btn_edge_detection.text()).width() + 20)

        self.file_selection_widget = QWidget(self)
        self.file_selection_layout = QVBoxLayout()
        self.file_selection_widget.setLayout(self.file_selection_layout)

        self.layout.addWidget(self.file_selection_widget)
        self.file_selection_widget.hide()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.file_selection_layout.addWidget(self.image_label)

        self.btn_upload = QPushButton('Upload Gambar', self)
        self.btn_upload.clicked.connect(self.loadImage)
        self.btn_upload.setFixedWidth(self.btn_upload.fontMetrics().boundingRect(self.btn_upload.text()).width() + 20)

        self.image_box = QLabel(self)
        self.image_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.file_selection_layout.addWidget(self.image_box)

        self.file_selection_layout.addWidget(self.btn_upload)

        self.btn_convert = QPushButton('Konversi', self)
        self.btn_convert.clicked.connect(self.performHistogramEqualization)
        self.btn_convert.setFixedWidth(self.btn_convert.fontMetrics().boundingRect(self.btn_convert.text()).width() + 20)
        self.layout.addWidget(self.btn_convert)
        self.btn_convert.hide()

        self.btn_home = QPushButton('Home', self)
        self.btn_home.clicked.connect(self.showHomePage)
        self.btn_home.setFixedWidth(self.btn_home.fontMetrics().boundingRect(self.btn_home.text()).width() + 20)
        self.layout.addWidget(self.btn_home)
        self.btn_home.hide()

        self.image_original = None
        self.image_equalized = None
        self.image_edges = None

    def showFileSelectionPage(self):
        self.btn_equalize.hide()
        self.btn_edge_detection.hide()
        self.file_selection_widget.show()

    def showHomePage(self):
        self.btn_equalize.show()
        self.btn_edge_detection.show()
        self.file_selection_widget.hide()
        self.image_label.clear()
        self.btn_home.hide()
        self.btn_convert.hide()
        self.image_original = None
        self.image_equalized = None
        self.image_edges = None

    def loadImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_name, _ = QFileDialog.getOpenFileName(self, 'Pilih Gambar', '', 'Images (*.png *.xpm *.jpg *.bmp *.jpeg *.tiff);;All Files (*)', options=options)

        if file_name:
            self.image_original = cv2.imread(file_name)
            if self.image_original is not None:
                image_resized = self.resizeImageToFitLabel(self.image_original, self.image_box)
                self.displayImage(image_resized)
                self.btn_convert.show()
                self.btn_home.show()
                self.btn_edge_detection.show()

    def resizeImageToFitLabel(self, image, label):
        label_size = label.size()
        image_height, image_width, _ = image.shape
        label_width = label_size.width()
        label_height = label_size.height()

        if label_width / label_height > image_width / image_height:
            scaled_width = label_width
            scaled_height = int(label_width / image_width * image_height)
        else:
            scaled_height = label_height
            scaled_width = int(label_height / image_height * image_width)

        return cv2.resize(image, (scaled_width, scaled_height))

    def displayImage(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def performHistogramEqualization(self):
        if self.image_original is not None:
            image_equalized = self.equalizeHistogramRGB(self.image_original)
            self.image_equalized = image_equalized

            image_original_rgb = cv2.cvtColor(self.image_original, cv2.COLOR_BGR2RGB)
            image_equalized_rgb = cv2.cvtColor(image_equalized, cv2.COLOR_BGR2RGB)

            combined_image = np.hstack([image_original_rgb, image_equalized_rgb])

            plt.figure(figsize=(12, 6), clear=True)  # Menghapus gambar sebelumnya

            plt.subplot(231)
            plt.imshow(image_equalized_rgb)
            plt.title('Hasil Equalisasi Histogram')
            plt.axis('off')

            hist_r = cv2.calcHist([image_equalized], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image_equalized], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image_equalized], [2], None, [256], [0, 256])

            plt.subplot(232)
            plt.plot(hist_r, color='r')
            plt.title('Histogram Kanal R')
            plt.xlim([0, 256])

            plt.subplot(233)
            plt.plot(hist_g, color='g')
            plt.title('Histogram Kanal G')
            plt.xlim([0, 256])

            plt.subplot(234)
            plt.plot(hist_b, color='b')
            plt.title('Histogram Kanal B')
            plt.xlim([0, 256])

            plt.subplot(235)
            plt.plot(hist_r, color='r', label='R')
            plt.plot(hist_g, color='g', label='G')
            plt.plot(hist_b, color='b', label='B')
            plt.title('Histogram Gabungan R,G,B')
            plt.xlim([0, 256])
            plt.legend()

            plt.tight_layout()
            plt.show()

    def equalizeHistogramRGB(self, image):
        r, g, b = cv2.split(image)
        r_eq = cv2.equalizeHist(r)
        g_eq = cv2.equalizeHist(g)
        b_eq = cv2.equalizeHist(b)
        equalized_image = cv2.merge((r_eq, g_eq, b_eq))
        return equalized_image

    def performEdgeDetection(self):
        if self.image_original is not None:
            image_edges = self.detectEdges(self.image_original)
            self.image_edges = image_edges

            image_original_rgb = cv2.cvtColor(self.image_original, cv2.COLOR_BGR2RGB)

            combined_image = np.hstack([image_original_rgb, image_edges])

            self.displayImage(combined_image)
            self.btn_home.show()

    def detectEdges(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return edges_rgb

def main():
    app = QApplication(sys.argv)
    ex = HistogramEqualizationApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
