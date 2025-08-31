import asyncio

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import cv2
import numpy as np
import re
import logging
import os

from services.ocr_service import OCRService
from services.image_processor import ImageProcessor
from services.text_extractor import TextExtractor
from concurrent.futures import ThreadPoolExecutor


# Initialize FastAPI app
app = FastAPI(title="OCR Service Demo (PaddleOCR)")
executor = ThreadPoolExecutor(max_workers=2)

# Initialize services
ocr_service = OCRService()
image_processor = ImageProcessor()
text_extractor = TextExtractor()

@app.post("/process")
async def process(file: UploadFile = File(...)):
    try:
        # Read and save uploaded file
        loop = asyncio.get_event_loop()
        contents = await file.read()

        # original_path = "temp_original.jpg"
        # with open(original_path, "wb") as f:
        #     f.write(contents)

        # Decode and validate image
        original_image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if original_image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        image_processor.validate_image(original_image)

        # Process image (resize if needed and run OCR)
        # image = cv2.imread(original_path)
        # result = image_processor.process_image(original_image, ocr_service.ocr)
        result = await loop.run_in_executor(executor, lambda: image_processor.process_image(original_image, ocr_service.ocr))

        # Save results
        for idx, res in enumerate(result):
            img_path = os.path.join(ocr_service.output_dir, "preprocessed.png")
            json_path = os.path.join(ocr_service.output_dir, "result.json")
            res.save_to_img(img_path)
            res.save_to_json(json_path)

        # Extract texts and perform line grouping
        rec_boxes = result[0]['rec_boxes']
        process_img = cv2.imread(img_path)
        cropped_results, final_texts = ocr_service.crop_and_predict(process_img, rec_boxes)

        if cropped_results:
            df_boxes = ocr_service.create_boxes_dataframe(cropped_results)
            line_df = ocr_service.group_boxes_to_lines(df_boxes)
            line_results = [(idx, row['line_text'], row['bbox']) for idx, row in line_df.iterrows()]
            # Extract text from line results using attribute access
            extract_text = "\n".join(row.line_text for row in line_df.itertuples())
        else:
            line_results = []
            extract_text = ""

        # Extract specific information


        results = {
            "date": text_extractor.find_date(extract_text),
            "total": text_extractor.find_total(extract_text),
            "address": text_extractor.find_address(extract_text),
            "phone": text_extractor.find_phone(extract_text),
            "raw_text": extract_text,
        }

        return JSONResponse(content=results)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
