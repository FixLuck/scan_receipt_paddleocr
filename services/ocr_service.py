from paddleocr import PaddleOCR
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from torch import nn
import torch
import cv2
import pandas as pd
from PIL import Image
import os

class OCRService:
    def __init__(self):
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.VIETOCR_MODEL_PATH = r'C:\Users\caube\AppData\Local\Temp\vgg_seq2seq.pth'
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize transformer model
        self.tran = nn.Transformer(batch_first=True)

        # Initialize VietOCR
        config = self.load_vietocr_config()
        self.predictor = Predictor(config)

        # Initialize PaddleOCR
        self.ocr = PaddleOCR(use_textline_orientation=True, lang='vi')

    def load_vietocr_config(self):
        """Load and configure VietOCR model."""
        config = Cfg.load_config_from_name('vgg_seq2seq')
        config['weights'] = self.VIETOCR_MODEL_PATH
        config['device'] = self.DEVICE
        return config

    def crop_and_predict(self, process_img, rec_boxes):
        """Crop image regions and predict text using VietOCR."""
        cropped_results = []
        final_texts = []

        for box_idx, rec_box in enumerate(rec_boxes):
            x_min, y_min, x_max, y_max = rec_box

            # Validate coordinates
            if not (0 <= y_min < y_max <= process_img.shape[0] and 0 <= x_min < x_max <= process_img.shape[1]):
                print(f"Invalid box coordinates at index {box_idx}: {rec_box}")
                continue

            # Crop and predict
            crop_img = process_img[y_min:y_max, x_min:x_max]
            pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            text = self.predictor.predict(pil_img)

            # Store results
            cropped_results.append({
                'crop_img': crop_img,
                'predicted_text': text,
                'box': rec_box,
                'box_idx': box_idx
            })
            final_texts.append(f"[{box_idx}] {text}")
            # print(f"Crop {box_idx}: {text}")

        return cropped_results, final_texts

    def create_boxes_dataframe(self, cropped_results):
        """Create DataFrame from cropped results."""
        return pd.DataFrame([{
            'x1': res['box'][0],
            'y1': res['box'][1],
            'x2': res['box'][2],
            'y2': res['box'][3],
            'text_vietocr': res['predicted_text']
        } for res in cropped_results])

    def group_boxes_to_lines(self, df, line_threshold=30):
        """Group text boxes into lines based on y-coordinate proximity."""
        df_sorted = df.sort_values(by='y1').reset_index(drop=True)
        lines, cur_line, last_y = [], [], -1000

        for _, row in df_sorted.iterrows():
            y_center = (row['y1'] + row['y2']) / 2
            if abs(y_center - last_y) > line_threshold:
                if cur_line:
                    lines.append(cur_line)
                cur_line = [row]
                last_y = y_center
            else:
                cur_line.append(row)

        if cur_line:
            lines.append(cur_line)

        # Combine text and get line bounding box
        line_results = []
        for line in lines:
            line_text = ' '.join([r['text_vietocr'] for r in sorted(line, key=lambda r: r['x1'])])
            x1 = min(r['x1'] for r in line)
            y1 = min(r['y1'] for r in line)
            x2 = max(r['x2'] for r in line)
            y2 = max(r['y2'] for r in line)
            line_results.append({'line_text': line_text, 'bbox': [x1, y1, x2, y2]})

        return pd.DataFrame(line_results)