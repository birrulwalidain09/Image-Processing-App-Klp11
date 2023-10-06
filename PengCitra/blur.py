import cv2

class FaceBlurrer:
    def __init__(self):
        self.blur_intensity = 0

    def set_blur_intensity(self, intensity):
        self.blur_intensity = intensity

    def apply_face_blur(self, image_path):
        image = cv2.imread(image_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_roi = image[y:y+h, x:x+w]
            # Terapkan efek blur pada wajah
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
            image[y:y+h, x:x+w] = blurred_face
        
        return image
