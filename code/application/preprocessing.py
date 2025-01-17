import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

class CuneiformProcessor:
    def __init__(self):
        dirname = os.path.dirname(__file__)
        #model_path = os.path.join(dirname, 'model')
        main_dir = os.path.abspath(os.path.join(dirname, '..', '..'))
        model_path = os.path.join(main_dir, 'model')
        loaded = tf.saved_model.load(model_path)
        self.model = loaded.signatures['serving_default']
        print("Model loaded successfully.")
        print("Model Output Keys:", loaded.signatures['serving_default'].structured_outputs)
       
        self.letters = {
            0: 'aa', 1: 'ba', 2: 'cha', 3: 'da', 4: 'de', 5: 'do', 6: 'ee', 7: 'fa',
            8: 'ga', 9: 'go', 10: 'ha', 11: 'ja', 12: 'je', 13: 'ka', 14: 'kha', 15: 'ko',
            16: 'la', 17: 'ma', 18: 'me', 19: 'mo', 20: 'na', 21: 'no', 22: 'oo', 23: 'pa',
            24: 'ra', 25: 'ro', 26: 'sa', 27: '-', 28: 'sha', 29: 'ta', 30: 'tha', 31: 'thra',
            32: 'to', 33: 'va', 34: 've', 35: 'ya', 36: 'za'
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def resize_and_pad(self, image, target_size=(30, 30)):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        h, w = image.shape
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        padded_image = np.ones((target_h, target_w), dtype=np.uint8) * 255

        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = scaled_image

        return padded_image

    def process_image(self, image_array):
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV) 

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
        threshed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)

        contours, _ = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

        for i, contour in enumerate(sorted_contours):
            x, y, w, h = cv2.boundingRect(contour)
            if w > 0.5 * image_array.shape[1] and h > 0.5 * image_array.shape[0]:
                new_image = image_array[y + 10:y + h - 10, x + 10:x + w - 10]
                
                self._process_inner_contours(new_image)

        return image_array 

    def _process_inner_contours(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        threshed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)

        contours, _ = cv2.findContours(threshed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
        for contour in sorted_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < 0.5 * image.shape[0] and w < 0.5 * image.shape[1]:
                letter_image = image[y - 3:y + h + 3, x - 3:x + w + 3]
                letter_image = cv2.cvtColor(letter_image, cv2.COLOR_BGR2GRAY)
                resized = self.resize_and_pad(letter_image)
                resized = 255 - resized
                resized = resized / 255.0
                resized = resized.reshape( 30, 30, 1)

                # Convert the input to a tensor
                input_tensor = tf.convert_to_tensor(resized, dtype=tf.float32)
                
                input_tensor = tf.expand_dims(input_tensor, axis=0)  # Add batch dimension
                
                # Use the model's serving_default signature for prediction
                predictions = self.model(input_tensor)
                
                # Extract the prediction result
                prediction = predictions['output_0']  # Adjust the key based on your model's output layer name
                label_index = np.argmax(prediction) 
                #if np.max(prediction) 
                #> 0.7 else None
                if(label_index >= 0  and label_index < 37 and len(contour) > 10):
                    cv2.putText(image, self.letters[label_index], (x - 2 , y - 3 ), self.font, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    def predict(self, image_array):
        processed_image = self.process_image(image_array)
        return processed_image

