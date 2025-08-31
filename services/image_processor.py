import cv2
import numpy as np
import logging

class ImageProcessor:
    def validate_image(self, img):
        """Validate image properties."""
        if img is None:
            raise ValueError("Image is required")

        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError("Image must be BGR (OpenCV format)")

        h, w = img.shape[:2]
        if h < 100 or w < 100:
            raise ValueError("Image must be at least 100x100")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if np.mean(gray) > 250:
            logging.warning("Warning: Image appears to be mostly blank")

        if np.std(gray) < 10:
            logging.warning("Image has low contrast, OCR may fail")

        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian < 100:
            logging.warning("Image appears blurry, OCR may be less accurate")

        return True
    # def process_image(self, image, ocr):
    #     """Process image with resizing if needed and run OCR."""
    #     height, width = image.shape[:2]
    #     if height > 1000:
    #         scale_percent = 50
    #         width = int(width * scale_percent / 100)
    #         height = int(height * scale_percent / 100)
    #         resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    #         cv2.imwrite('resized_image.jpg', resized_image)
    #         return ocr.predict(resized_image)
    #     return ocr.predict(image)

    def process_image(self, image, ocr):
        """Process image with resizing if needed and run OCR."""
        height, width = image.shape[:2]
        if height > 1000:
            scale_percent = 50
            width = int(width * scale_percent / 100)
            height = int(height * scale_percent / 100)
            resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            return ocr.predict(resized_image)
        return ocr.predict(image)