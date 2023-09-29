import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QSizePolicy, QStackedWidget, QMessageBox, QSlider
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt
from PIL import Image, ImageQt
from PyQt5.QtWidgets import QPushButton

 
class HistogramEqualizationApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.createHomePage()  # Membuat halaman awal
        self.createSegmentationPage()
        
        self.segmentation_image = None

    def initUI(self):
        self.setWindowTitle('Aplikasi Pengolahan Citra_Kelompok 11')
        self.setGeometry(100, 100, 800, 600)
        self.setFixedSize(800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
 
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.stacked_widget = QStackedWidget(self)
        self.layout.addWidget(self.stacked_widget)

        self.home_page = QWidget(self)
        self.stacked_widget.addWidget(self.home_page)

        self.image_page = QWidget(self)
        self.stacked_widget.addWidget(self.image_page)

        self.face_blurring_page = QWidget(self)
        self.stacked_widget.addWidget(self.face_blurring_page)
         

        self.home_layout = QVBoxLayout()
        self.home_page.setLayout(self.home_layout)

        self.image_layout = QVBoxLayout()
        self.image_page.setLayout(self.image_layout)

        self.face_blurring_layout = QVBoxLayout()
        self.face_blurring_page.setLayout(self.face_blurring_layout)

        self.btn_equalize = QPushButton('Histogram Equalization', self)
        self.home_layout.addWidget(self.btn_equalize)
        self.btn_equalize.clicked.connect(self.showImagePage)
        self.btn_equalize.setFixedWidth(self.btn_equalize.fontMetrics().boundingRect(self.btn_equalize.text()).width() + 20)

        self.btn_edge_detection = QPushButton('Edge Detection', self)
        self.home_layout.addWidget(self.btn_edge_detection)
        self.btn_edge_detection.clicked.connect(self.showEdgeDetectionPage)
        self.btn_edge_detection.setFixedWidth(self.btn_edge_detection.fontMetrics().boundingRect(self.btn_edge_detection.text()).width() + 20)

        self.btn_face_blurring = QPushButton('Face Blurring', self)
        self.home_layout.addWidget(self.btn_face_blurring)
        self.btn_face_blurring.clicked.connect(self.showFaceBlurringPage)
        self.btn_face_blurring.setFixedWidth(self.btn_face_blurring.fontMetrics().boundingRect(self.btn_face_blurring.text()).width() + 20)

        self.btn_upload = QPushButton('Upload Gambar', self)
        self.image_layout.addWidget(self.btn_upload)
        self.btn_upload.clicked.connect(self.loadImage)

        self.image_container = QLabel(self)
        self.image_container.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_container.setStyleSheet("background-color: white; border: 1px solid black;")
        self.image_layout.addWidget(self.image_container)

        self.btn_convert = QPushButton('Konversi', self)
        self.image_layout.addWidget(self.btn_convert)
        self.btn_convert.clicked.connect(self.performHistogramEqualization)
        self.btn_convert.setFixedWidth(self.btn_convert.fontMetrics().boundingRect(self.btn_convert.text()).width() + 20)
        self.btn_convert.hide()
        
        self.btn_upload_face = QPushButton('Upload Gambar', self)   
        self.face_blurring_layout.addWidget(self.btn_upload_face)
        self.btn_upload_face.clicked.connect(self.loadImageFaceBlurring)
        
        self.btn_blur_face = QPushButton('Blurring', self)
        self.face_blurring_layout.addWidget(self.btn_blur_face)
        self.btn_blur_face.clicked.connect(self.performFaceBlurring)  # Connect the button to performFaceBlurring
        self.btn_blur_face.hide()
        
        self.btn_segmentation = QPushButton('Segmentation', self)  # Tambahkan tombol Segmentation
        self.home_layout.addWidget(self.btn_segmentation)
        self.btn_segmentation.clicked.connect(self.showSegmentationPage)
        self.btn_segmentation.setFixedWidth(self.btn_segmentation.fontMetrics().boundingRect(self.btn_segmentation.text()).width() + 20)

        self.btn_home_face = QPushButton('Home', self)
        self.face_blurring_layout.addWidget(self.btn_home_face)
        self.btn_home_face.clicked.connect(self.goToHomePage)
        self.btn_home_face.setFixedWidth(self.btn_home_face.fontMetrics().boundingRect(self.btn_home_face.text()).width() + 20)
        self.btn_home_face.hide()

        self.btn_home = QPushButton('Home', self)
        self.image_layout.addWidget(self.btn_home)
        self.btn_home.clicked.connect(self.resetApp)
        self.btn_home.setFixedWidth(self.btn_home.fontMetrics().boundingRect(self.btn_home.text()).width() + 20)
        self.btn_home.hide()

        self.image_original = None
        self.image_equalized = None

        # Slider untuk mengatur tingkat blurring wajah
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setMinimum(1)
        self.scale_slider.setMaximum(100)
        self.scale_slider.setValue(30)  # Nilai awal tingkat blurring
        self.scale_slider.valueChanged.connect(self.updateBlurScale)
        self.scale_slider.hide()

        self.scale_label = QLabel('Scale: 30%', self)
        self.scale_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scale_label.hide()

        self.face_image_container = QLabel(self)
        self.face_image_container.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.face_image_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.face_image_container.setStyleSheet("background-color: white; border: 1px solid black;")
        self.face_blurring_layout.addWidget(self.face_image_container)
        self.face_blurring_layout.addWidget(self.scale_slider)
        self.face_blurring_layout.addWidget(self.scale_label)

        self.btn_blur_face = QPushButton('Blurring', self)
        self.face_blurring_layout.addWidget(self.btn_blur_face)
        self.btn_blur_face.clicked.connect(self.performFaceBlurring)
        self.btn_blur_face.hide()
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        
    def createHomePage(self):
        self.home_page = QWidget(self)
        self.stacked_widget.addWidget(self.home_page)

        self.home_layout = QVBoxLayout()
        self.home_page.setLayout(self.home_layout)

        self.btn_equalize = QPushButton('Histogram Equalization', self)
        self.home_layout.addWidget(self.btn_equalize)
        self.btn_equalize.clicked.connect(self.showImagePage)
        self.btn_equalize.setFixedWidth(self.btn_equalize.fontMetrics().boundingRect(self.btn_equalize.text()).width() + 20)

        self.btn_edge_detection = QPushButton('Edge Detection', self)
        self.home_layout.addWidget(self.btn_edge_detection)
        self.btn_edge_detection.clicked.connect(self.showEdgeDetectionPage)
        self.btn_edge_detection.setFixedWidth(self.btn_edge_detection.fontMetrics().boundingRect(self.btn_edge_detection.text()).width() + 20)

        self.btn_face_blurring = QPushButton('Face Blurring', self)
        self.home_layout.addWidget(self.btn_face_blurring)
        self.btn_face_blurring.clicked.connect(self.showFaceBlurringPage)
        self.btn_face_blurring.setFixedWidth(self.btn_face_blurring.fontMetrics().boundingRect(self.btn_face_blurring.text()).width() + 20)
        
        self.btn_segmentation = QPushButton('Segmentation', self)
        self.home_layout.addWidget(self.btn_segmentation)
        self.btn_segmentation.clicked.connect(self.showSegmentationPage)
        self.btn_segmentation.setFixedWidth(self.btn_segmentation.fontMetrics().boundingRect(self.btn_segmentation.text()).width() + 20)

    def goToHomePage(self):
        self.stacked_widget.setCurrentWidget(self.home_page)
        self.btn_upload.hide()
        self.btn_blur_face.hide()
        self.scale_slider.hide()
        self.scale_label.hide()
        self.btn_home_face.hide()

          
    def loadImageFaceBlurring(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_name, _ = QFileDialog.getOpenFileName(self, 'Pilih Gambar', '', 'Images (*.png *.xpm *.jpg *.bmp *.jpeg *.tiff);;All Files (*)', options=options)

        if file_name:
            self.image_original = cv2.imread(file_name)
            if self.image_original is not None:
                self.btn_blur_face.show()
                self.scale_slider.show()
                self.scale_label.show()
                image_resized = self.resizeImageToFitLabel(self.image_original, self.face_image_container)
                self.displayImageFaceBlurring(image_resized)


    def loadImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_name, _ = QFileDialog.getOpenFileName(self, 'Pilih Gambar', '', 'Images (*.png *.xpm *.jpg *.bmp *.jpeg *.tiff);;All Files (*)', options=options)

        if file_name:
            self.image_original = cv2.imread(file_name)
            if self.image_original is not None:
                self.btn_convert.show()
                self.btn_home.show()
                image_resized = self.resizeImageToFitLabel(self.image_original, self.image_container)
                self.displayImage(image_resized)


    def updateBlurScale(self):
        scale = self.scale_slider.value()
        self.scale_label.setText(f'Scale: {scale}%')

    def displayImage(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_container.setPixmap(pixmap)
        self.image_container.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def performHistogramEqualization(self):
        if self.image_original is not None:
            image_equalized = self.equalizeHistogramRGB(self.image_original)
            self.image_equalized = image_equalized

            image_original_rgb = cv2.cvtColor(self.image_original, cv2.COLOR_BGR2RGB)
            image_equalized_rgb = cv2.cvtColor(image_equalized, cv2.COLOR_BGR2RGB)

            combined_image = np.hstack([image_original_rgb, image_equalized_rgb])

            plt.figure(figsize=(12, 6), clear=True)

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

    def performFaceBlurring(self):
        if self.image_original is not None:
            gray_image = cv2.cvtColor(self.image_original, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) == 0:
                QMessageBox.information(self, 'Info', 'Tidak terdeteksi wajah.')
                return

            image_with_faces = self.image_original.copy()
            for (x, y, w, h) in faces:
                face = image_with_faces[y:y+h, x:x+w]
                scale = self.scale_slider.value() / 100.0
                kernel_size = int(min(w, h) * scale)
                if kernel_size % 2 == 0:
                    kernel_size += 1  # Pastikan ukuran kernel adalah ganjil
                face = cv2.GaussianBlur(face, (kernel_size, kernel_size), 30)  # Ganti ukuran kernel dan tingkat blurring sesuai kebutuhan
                image_with_faces[y:y+h, x:x+w] = face

            image_original_rgb = cv2.cvtColor(self.image_original, cv2.COLOR_BGR2RGB)
            image_with_faces_rgb = cv2.cvtColor(image_with_faces, cv2.COLOR_BGR2RGB)

            combined_image = np.hstack([image_original_rgb, image_with_faces_rgb])

            self.displayImageFaceBlurring(combined_image)
            self.btn_home.show()

    def displayImageFaceBlurring(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.face_image_container.setPixmap(pixmap)
        self.face_image_container.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def resetApp(self):
        # Mengatur semua variabel dan tampilan ke dalam kondisi awal
        self.image_original = None
        self.image_equalized = None
        self.image_edges = None
        self.image_container.clear()
        self.face_image_container.clear()
        self.btn_upload.hide()
        self.btn_convert.hide()
        self.btn_blur_face.hide()
        self.scale_slider.hide()
        self.scale_label.hide()
        self.btn_home.hide()
        self.stacked_widget.setCurrentWidget(self.home_page)

    def showImagePage(self):
        self.stacked_widget.setCurrentWidget(self.image_page)
        self.btn_upload.show()
        self.btn_convert.setText('Konversi')
        self.btn_convert.clicked.disconnect()
        self.btn_convert.clicked.connect(self.performHistogramEqualization)
        self.btn_home.show()


    def showEdgeDetectionPage(self):
        self.stacked_widget.setCurrentWidget(self.image_page)
        self.btn_upload.show()
        self.btn_convert.setText('Edge Detection')
        self.btn_convert.clicked.disconnect()
        self.btn_convert.clicked.connect(self.performEdgeDetection)
        self.btn_home.setText('Home')
        self.btn_home.clicked.disconnect()
        self.btn_home.clicked.connect(self.resetApp)
        self.image_container.clear()
        self.btn_upload.show()
        self.btn_convert.show()
        self.btn_home.show()

        # Mengganti nama button di bawah kotak gambar yang diupload
        self.btn_upload.setText('Upload Gambar')
        self.btn_convert.setText('Konversi')
    
    def showFaceBlurringPage(self):
        self.stacked_widget.setCurrentWidget(self.face_blurring_page)
        self.btn_upload_face.clicked.connect(self.loadImageFaceBlurring)
        self.btn_blur_face.show()
        self.btn_home_face.show()
        self.scale_slider.show()
        self.scale_label.show()
        
    def showSegmentationPage(self):
        self.stacked_widget.setCurrentWidget(self.segmentation_page)  # Menggunakan halaman yang telah dibuat sebelumnya
        self.btn_upload_segmentation.show()
        self.btn_segment.show()
        self.btn_home_segmentation.show()
        self.segmentation_image_container.clear()

        self.btn_upload_segmentation.setText('Upload Gambar')
        self.btn_upload_segmentation.clicked.disconnect()
        self.btn_upload_segmentation.clicked.connect(self.loadImageSegmentation)

        self.btn_segment.setText('Segmentasi')
        self.btn_segment.clicked.disconnect()
        self.btn_segment.clicked.connect(self.performSegmentation)



    def resizeImageToFitLabel(self, image, label):
        label_size = label.size()
        label_width = label_size.width()
        label_height = label_size.height()

        image_height, image_width, _ = image.shape  # Menggunakan tiga nilai yang dikembalikan oleh image.shape

        if label_width / label_height > image_width / image_height:
            scaled_width = label_width
            scaled_height = int(label_width / image_width * image_height)
        else:
            scaled_height = label_height
            scaled_width = int(label_height / image_height * image_width)

        return cv2.resize(image, (scaled_width, scaled_height))

    
    def createSegmentationPage(self):
        self.segmentation_page = QWidget(self)
        self.stacked_widget.addWidget(self.segmentation_page)

        self.segmentation_layout = QVBoxLayout()
        self.segmentation_page.setLayout(self.segmentation_layout)

        self.btn_upload_segmentation = QPushButton('Upload Gambar', self.segmentation_page)  # Mengakses objek btn_upload_segmentation melalui objek segmentation_page
        self.segmentation_layout.addWidget(self.btn_upload_segmentation)
        self.btn_upload_segmentation.clicked.connect(self.loadImageSegmentation)

        self.segmentation_image_container = QLabel(self.segmentation_page)  # Mengakses objek segmentation_image_container melalui objek segmentation_page
        self.segmentation_image_container.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.segmentation_image_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.segmentation_image_container.setStyleSheet("background-color: white; border: 1px solid black;")
        self.segmentation_layout.addWidget(self.segmentation_image_container)

        self.btn_segment = QPushButton('Segmentasi', self.segmentation_page)
        self.segmentation_layout.addWidget(self.btn_segment)
        self.btn_segment.clicked.connect(self.performSegmentation)
        self.btn_segment.setFixedWidth(self.btn_segment.fontMetrics().boundingRect(self.btn_segment.text()).width() + 20)
        self.btn_segment.hide()

        self.btn_home_segmentation = QPushButton('Home', self.segmentation_page)
        self.segmentation_layout.addWidget(self.btn_home_segmentation)
        self.btn_home_segmentation.clicked.connect(self.goToHomePage)
        self.btn_home_segmentation.setFixedWidth(self.btn_home_segmentation.fontMetrics().boundingRect(self.btn_home_segmentation.text()).width() + 20)
        self.btn_home_segmentation.hide()


    def goToSegmentationPage(self):
        self.stacked_widget.setCurrentWidget(self.segmentation_page)
        self.segmentation_page.btn_upload_segmentation.show()  # Mengakses btn_upload_segmentation melalui objek segmentation_page
        self.segmentation_page.btn_segment.show()
        self.segmentation_page.btn_home_segmentation.show()
        self.segmentation_page.segmentation_image_container.clear()


    def loadImageSegmentation(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        file_name, _ = QFileDialog.getOpenFileName(self, 'Pilih Gambar', '', 'Images (*.png *.xpm *.jpg *.bmp *.jpeg *.tiff);;All Files (*)', options=options)

        if file_name:
            self.segmentation_image = cv2.imread(file_name)
            if self.segmentation_image is None:
                QMessageBox.warning(self, 'Peringatan', 'Gagal membaca gambar. Pastikan file gambar valid.')
            else:
                self.btn_segment.show()
                image_resized = self.resizeImageToFitLabel(self.segmentation_image, self.segmentation_image_container)
                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)  # Konversi ke format warna RGB
                self.displaySegmentationImage(image_rgb)

    def displaySegmentationImage(self, image):
        height, width, _ = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.segmentation_image_container.setPixmap(pixmap)
        self.segmentation_image_container.setAlignment(Qt.AlignmentFlag.AlignCenter)



    def performSegmentation(self):
        if self.segmentation_image is not None:
            # Konversi citra ke format warna HSV (Hue, Saturation, Value)
            hsv_image = cv2.cvtColor(self.segmentation_image, cv2.COLOR_BGR2HSV)

            # Tentukan rentang warna untuk buah matang (merah)
            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])
            mask_red1 = cv2.inRange(hsv_image, lower_red, upper_red)

            lower_red = np.array([160, 100, 100])
            upper_red = np.array([180, 255, 255])
            mask_red2 = cv2.inRange(hsv_image, lower_red, upper_red)

            # Gabungkan hasil segmentasi
            segmentation_result = cv2.bitwise_or(mask_red1, mask_red2)

            # Temukan kontur objek yang tersegmentasi
            contours, _ = cv2.findContours(segmentation_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Loop melalui kontur dan tandai buah kopi yang matang dengan kotak dan label "matang"
            segmented_image = self.segmentation_image.copy()
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 10 and h > 10:  # Batasi ukuran objek yang akan ditandai
                    cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Tandai dengan kotak hijau
                    cv2.putText(segmented_image, 'Matang', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Tampilkan gambar original dan hasil segmentasi dalam satu gambar
            combined_image = np.hstack([self.segmentation_image, segmented_image])

            # Tampilkan gambar yang menggabungkan gambar original dan hasil segmentasi
            self.displaySegmentationImage(combined_image)



def main():
    app = QApplication(sys.argv)
    ex = HistogramEqualizationApp()
    ex.showFullScreen()  # Menampilkan aplikasi dalam mode fullscreen
    sys.exit(app.exec_())
    

if __name__ == '__main__':
    main()
 
