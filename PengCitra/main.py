from PyQt5.QtWidgets import *
from histogram import HistogramProcessor
from edge import EdgeProcessor
from blur import FaceBlurrer
from segmentation import SegmentationProcessor
from efek import apply_vector_effect
from PyQt5 import Qt
from PyQt5 import QtCore, QtGui 
from PyQt5.QtGui import *   
from PyQt5.QtCore import * 
from PyQt5.QtWidgets import QMessageBox
import matplotlib.pyplot as plt
import cv2
import sys 
import numpy as np
from PyQt5.QtGui import QImage, QPixmap


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Kelompok 11')
        self.Width = 800
        self.height = int(0.618 * self.Width)
        self.resize(self.Width, self.height)
        self.image_path = None
        
        self.histogram_processor = HistogramProcessor()
        self.edge_processor = EdgeProcessor()
        self.face_blurrer = FaceBlurrer()
        self.segmentation_processor = SegmentationProcessor(self.image_path)
        

        
        self.slider = None 
        self.image_uploaded = False
        

        # Left side buttons
        self.btn_1 = QPushButton('Upload Image', self)
        self.btn_2 = QPushButton('Histogram Equalization', self)
        self.btn_3 = QPushButton('Edge Detection', self)
        self.btn_4 = QPushButton('Face Blurring', self) 
        self.btn_5 = QPushButton('Segmentation', self)
        self.btn_6 = QPushButton('efek vektor', self)
        self.btn_1.setObjectName('left_button')
        self.btn_2.setObjectName('left_button')
        self.btn_3.setObjectName('left_button')
        self.btn_4.setObjectName('left_button')
        self.btn_5.setObjectName('left_button')
        self.btn_6.setObjectName('left_button')

        # Right side tabs
        self.right_widget = QTabWidget()

        self.initUI()

    def initUI(self):
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.btn_1)
        left_layout.addWidget(self.btn_2)
        left_layout.addWidget(self.btn_3)
        left_layout.addWidget(self.btn_4)
        left_layout.addWidget(self.btn_5)
        left_layout.addWidget(self.btn_6)
        left_layout.addStretch(1)
        left_layout.setSpacing(20)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setStyleSheet('''
            QPushButton{
                border:none;
                color:rgb(0,0,0);
                font-size:20px;
                font-weight:400;
                text-align:left;
            }
            QPushButton#left_button:hover{
                font-weight:600;
                background:rgb(220,220,220);
                border-left:5px solid blue;
            }
            QWidget#left_widget{
                background:rgb(220,220,220);
                border-top:1px solid white;
                border-bottom:1px solid white;
                border-left:1px solid white;
                border-top-left-radius:10px;
                border-bottom-left-radius:10px;
            }
        ''')

        main_layout = QHBoxLayout()
        main_layout.addWidget(left_widget)
        main_layout.addWidget(self.right_widget)
        main_layout.setStretch(0, 40)
        main_layout.setStretch(1, 200)
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        

        # Menambahkan fungsi untuk masing-masing tombol
        self.btn_1.clicked.connect(lambda: self.switch_to_tab(self.btn_1))
        self.btn_2.clicked.connect(lambda: self.switch_to_tab(self.btn_2))
        self.btn_3.clicked.connect(lambda: self.switch_to_tab(self.btn_3))
        self.btn_4.clicked.connect(lambda: self.switch_to_tab(self.btn_4))
        self.btn_5.clicked.connect(lambda: self.switch_to_tab(self.btn_5))
        self.btn_6.clicked.connect(lambda: self.switch_to_tab(self.btn_6))

    def switch_to_tab(self, button):
        # Menghapus semua tab yang ada di right_widget
        self.right_widget.clear()

        # Membuat tab baru sesuai dengan tombol yang ditekan
        if button == self.btn_1:
            tab_widget = self.create_upload_tab()
        elif button == self.btn_2:
            tab_widget = self.create_histogram_tab()
        elif button == self.btn_3:
            tab_widget = self.create_edge_detection_tab()
        elif button == self.btn_4:
            tab_widget = self.create_face_blurring_tab()
        elif button == self.btn_5:
            tab_widget = self.create_segmentation_tab()
        elif button == self.btn_6:
            tab_widget = self.create_efek_vektor_tab()

        # Menambahkan tab baru ke right_widget
        self.right_widget.addTab(tab_widget, button.text())

    def create_upload_tab(self):    
        upload_tab = QWidget()

        # Membuat layout untuk tab "Upload Image"
        upload_layout = QVBoxLayout(upload_tab)

        # Membuat tombol "Upload" dan "Reset"
        upload_button = QPushButton("Upload", upload_tab)
        reset_button = QPushButton("Reset", upload_tab)

        # Mengatur ukuran tombol sesuai dengan teks di dalamnya
        upload_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        reset_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        # Menambahkan fungsi untuk menghandle klik tombol "Upload"
        upload_button.clicked.connect(self.upload_image)
        # Menambahkan fungsi untuk menghandle klik tombol "Reset"
        reset_button.clicked.connect(self.reset_image)

        # Menempatkan tombol di sudut kiri atas pada right_layout tab "Upload Image"
        button_layout = QHBoxLayout()
        button_layout.addWidget(upload_button)
        button_layout.addWidget(reset_button)

        upload_layout.addLayout(button_layout)

        self.image_label = QLabel(upload_tab)  # Label untuk menampilkan gambar
        upload_layout.addWidget(self.image_label)

        return upload_tab

    
    def upload_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Upload Image", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)
        if file_name:
            # Menyimpan path gambar yang diupload oleh pengguna
            if file_name.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg')):
                self.image_path = file_name
                # Menampilkan gambar pada label
                pixmap = QPixmap(file_name)
                self.image_label.setPixmap(pixmap)
                self.image_label.setAlignment(Qt.AlignCenter)  # Gambar akan ditampilkan di tengah label
                self.image_uploaded = True  # Atur status gambar telah diunggah
            else:
                QMessageBox.warning(self, 'Peringatan', 'Format gambar tidak valid. Harap unggah gambar pohon kopi dengan buahnya.')
        else:
            QMessageBox.warning(self, 'Peringatan', 'Harap unggah gambar terlebih dahulu!')


    def reset_image(self):
        # Menghapus gambar yang ditampilkan pada label
        self.image_label.clear()
        
    def create_histogram_tab(self):
        histogram_tab = QWidget()
        layout = QVBoxLayout(histogram_tab)

        # Tambahkan tombol "Apply Histogram Equalization"
        apply_histogram_button = QPushButton("Apply Histogram Equalization", histogram_tab)
        apply_histogram_button.clicked.connect(self.apply_histogram_equalization)
        layout.addWidget(apply_histogram_button)

        # Tambahkan widget processor ke dalam layout
        layout.addWidget(self.histogram_processor)
        return histogram_tab
    
    def apply_histogram_equalization(self):
        # Memanggil metode untuk memproses gambar menggunakan histogram equalization
         self.histogram_processor.apply_histogram_equalization(self.image_path)

    def create_edge_detection_tab(self):
        edge_detection_tab = QWidget()
        layout = QVBoxLayout(edge_detection_tab)

        # Tambahkan tombol "Apply Edge Detection"
        apply_edge_button = QPushButton("Apply Edge Detection", edge_detection_tab)
        apply_edge_button.clicked.connect(self.apply_edge_detection)
        layout.addWidget(apply_edge_button)

        # Tambahkan widget processor ke dalam layout
        layout.addWidget(self.edge_processor)
        return edge_detection_tab

    def apply_edge_detection(self):
        # Memanggil metode untuk memproses gambar menggunakan edge detection
        self.edge_processor.apply_edge_detection(self.image_path)

    def create_face_blurring_tab(self):
        face_blurring_tab = QWidget()
        layout = QVBoxLayout(face_blurring_tab)

        # Tambahkan tombol "Apply Face Blur"
        apply_blur_button = QPushButton("Apply Face Blur", face_blurring_tab)
        apply_blur_button.clicked.connect(self.apply_face_blur)
        layout.addWidget(apply_blur_button)

        # Tambahkan slider untuk mengatur tingkat blur
        self.slider = QSlider(Qt.Horizontal, face_blurring_tab)  # Gunakan self.slider agar bisa diakses di metode lain
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_blur_intensity)
        layout.addWidget(self.slider)

        # Buat widget untuk menampilkan hasil face blurring
        self.face_blurring_result_widget = QLabel(face_blurring_tab)
        layout.addWidget(self.face_blurring_result_widget)

        return face_blurring_tab


    def apply_face_blur(self):
        # Pastikan ada gambar yang diupload sebelum mengaplikasikan face blur
        if hasattr(self, 'image_path') and self.image_path:
            # Panggil metode apply_face_blur dari kelas FaceBlurrer
            gambar_terabur = self.face_blurrer.apply_face_blur(self.image_path)

            # Konversi gambar hasil blur ke format RGB
            gambar_terabur_rgb = cv2.cvtColor(gambar_terabur, cv2.COLOR_BGR2RGB)

            # Konversi gambar ke format QImage agar bisa ditampilkan di QLabel
            height, width, channel = gambar_terabur_rgb.shape
            bytes_per_line = 3 * width
            q_img = QImage(gambar_terabur_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            # Tampilkan gambar terabur di self.face_blurring_result_widget
            self.face_blurring_result_widget.setPixmap(pixmap)
        else:
            QMessageBox.warning(self, 'Peringatan', 'Silakan upload gambar terlebih dahulu!')


    def update_blur_intensity(self):
        # Mendapatkan nilai slider untuk intensitas blur
        blur_intensity = self.slider.value()
        # Mengatur intensitas blur pada objek face_blurrer
        self.face_blurrer.set_blur_intensity(blur_intensity)
    
        # Menggunakan fungsi apply_face_blur dengan intensitas blur yang diperbarui
        self.apply_face_blur()


    def create_segmentation_tab(self):
        segmentation_tab = QWidget()
        layout = QVBoxLayout(segmentation_tab)

        # Tambahkan tombol "Apply Segmentation"
        apply_segmentation_button = QPushButton("Apply Segmentation", segmentation_tab)
        apply_segmentation_button.clicked.connect(self.apply_segmentation)  # Hubungkan dengan metode apply_segmentation
        layout.addWidget(apply_segmentation_button)

        # Tambahkan label untuk menampilkan hasil segmentasi
        self.segmentation_result_label = QLabel(self)
        layout.addWidget(self.segmentation_result_label)

        return segmentation_tab

    def apply_segmentation(self):
        if self.image_path:
            # Baca gambar dari path yang disimpan di self.image_path
            original_image = cv2.imread(self.image_path)

            if original_image is None:
                # Menampilkan pesan kesalahan jika gagal membaca gambar
                QMessageBox.warning(self, 'Peringatan', 'Gagal membaca gambar. Pastikan format gambar valid.')
                return
        
            # Konversi gambar ke format Lab
            lab_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2Lab)
        
            # Tentukan batasan warna untuk segmentasi (merah untuk buah matang, hijau untuk buah belum matang)
            lower_red = np.array([0, 120, 70])
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
            
            # Temukan kontur buah kopi yang sudah matang
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Loop melalui setiap kontur dan tambahkan kotak di sekitar buah kopi yang sudah matang
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 10 and h > 10:  # Batasi ukuran objek yang akan ditandai
                    cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Tandai dengan kotak hijau
                    cv2.putText(segmented_image, 'Matang', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


            if np.any(segmented_image):
                # Konversi gambar hasil segmentasi ke format RGB
                segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

                # Konversi gambar ke format QImage agar bisa ditampilkan di QLabel
                height, width, channel = segmented_image_rgb.shape
                bytes_per_line = 3 * width
                q_img = QImage(segmented_image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.segmentation_result_label.setPixmap(pixmap)
            else:
                # Menampilkan pesan kesalahan jika objek tidak ditemukan
                QMessageBox.warning(self, 'Peringatan', 'Objek tidak ditemukan dalam gambar. Sesuaikan nilai batas warna.')
                self.segmentation_result_label.clear()  # Menghapus gambar hasil segmentasi sebelumnya
        else:
            # Menampilkan pesan kesalahan jika gambar belum diunggah
            QMessageBox.warning(self, 'Peringatan', 'Silakan upload gambar terlebih dahulu!')
            self.segmentation_result_label.clear()  # Menghapus gambar hasil segmentasi sebelumnya


    def create_efek_vektor_tab(self):
        efek_vektor_tab = QWidget()

        # Tambahkan tombol "Apply Efek Vektor"
        apply_vector_effect_button = QPushButton("Apply Efek Vektor", efek_vektor_tab)
        apply_vector_effect_button.clicked.connect(self.apply_vector_effect)
        layout = QVBoxLayout(efek_vektor_tab)
        layout.addWidget(apply_vector_effect_button)

        # Tambahkan label untuk menampilkan hasil efek vektor
        self.vector_effect_result_label = QLabel(self)
        layout.addWidget(self.vector_effect_result_label)

        return efek_vektor_tab

    def apply_vector_effect(self):
        if self.image_path:
            # Tentukan path untuk menyimpan gambar hasil efek vektor
            output_path = "vector_effect_output.jpg"

            # Panggil fungsi apply_vector_effect dari efek.py
            output_image_path = apply_vector_effect(self.image_path, output_path)

            # Tampilkan gambar hasil efek vektor di self.vector_effect_result_label
            pixmap = QPixmap(output_image_path)
            self.vector_effect_result_label.setPixmap(pixmap)
        else:
            # Menampilkan pesan kesalahan jika gambar belum diunggah
            QMessageBox.warning(self, 'Peringatan', 'Silakan upload gambar terlebih dahulu!')
            self.vector_effect_result_label.clear()  # Menghapus gambar hasil efek vektor sebelumnya
       
       
            
            
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    sys.exit(app.exec_())
