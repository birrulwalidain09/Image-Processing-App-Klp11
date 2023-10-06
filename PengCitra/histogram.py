from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QImage, QColor, qRgb,QPixmap
from PyQt5.QtCore import Qt

class HistogramProcessor(QWidget):
    def __init__(self, parent=None):
        super(HistogramProcessor, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

    def apply_histogram_equalization(self, image_path):
        original_image = QPixmap(image_path)
        image = original_image.toImage()

        # Process the image using histogram equalization algorithm here
        # For example, you can convert the image to grayscale and apply histogram equalization
        processed_image = self.histogram_equalization(image)

        # Display the processed image
        self.image_label.setPixmap(processed_image)

    def histogram_equalization(self, image):
        # Convert the image to grayscale
        gray_image = image.convertToFormat(QImage.Format_Grayscale8)

        # Compute histogram
        histogram = [0] * 256
        for y in range(gray_image.height()):
            for x in range(gray_image.width()):
                pixel_value = QColor(gray_image.pixel(x, y)).lightness()
                histogram[pixel_value] += 1

        # Compute cumulative distribution function (CDF)
        cdf = [sum(histogram[:i + 1]) for i in range(len(histogram))]

        # Normalize CDF
        cdf_normalized = [int((cdf[i] - cdf[0]) * 255 / (cdf[-1] - cdf[0])) for i in range(len(cdf))]

        # Apply histogram equalization
        equalized_image = QImage(gray_image.width(), gray_image.height(), QImage.Format_Grayscale8)
        for y in range(gray_image.height()):
            for x in range(gray_image.width()):
                pixel_value = QColor(gray_image.pixel(x, y)).lightness()
                new_pixel_value = cdf_normalized[pixel_value]
                equalized_image.setPixel(x, y, qRgb(new_pixel_value, new_pixel_value, new_pixel_value))

        return QPixmap.fromImage(equalized_image)