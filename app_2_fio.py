from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from io import BytesIO
import base64
import re
import cv2
import numpy as np
import easyocr
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tất cả các origin
    allow_credentials=True,
    allow_methods=["*"],  # tất cả các HTTP methods
    allow_headers=["*"],  # tất cả các headers
)

# Pydantic model để nhận base64 string từ client
class ImageRequest(BaseModel):
    image_base64: str


# phát hiện khu vực đỏ trong ảnh
def detect_red_area(image):
    """Detect and crop the red area in the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return image[y:y+h, x:x+w]
    return None


# đọc văn bản từ ảnh và phân tích giá trị số
def read_img(img):
    """Extract text from the image and parse for numeric values."""
    reader = easyocr.Reader(['en'], gpu=True)
    result = reader.readtext(img)
    
    if result:
        text = result[0][-2]
        cleaned_text = ''.join(re.findall(r'\d', text))
        number = int(cleaned_text) if cleaned_text else None
    else:
        text = ""
        number = None

    return text, number


# giải mã base64 thành ảnh
def decode_base64_image(base64_str):
    """Giải mã ảnh từ chuỗi base64"""
    # loại bỏ header nếu có
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    
    # thêm padding nếu thiếu
    missing_padding = len(base64_str) % 4
    if missing_padding:
        base64_str += '=' * (4 - missing_padding)

    # giải mã ảnh
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


@app.post("/process_image")
async def process_image(data: ImageRequest):
    """API to receive a base64 image and return extracted text and numbers."""
    try:
        base64_str = data.image_base64
        image = decode_base64_image(base64_str)

        # cropped_image = detect_red_area(image)

        text, number = read_img(image)
        return JSONResponse(content={
            'status': 'success',
            'text': text,
            'number': number
        })

    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid input data: Missing base64 string.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_html():
    # html
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>FastAPI</title>
    </head>
    <body>
        <h1>Hello world</h1>
    </body>
    </html>
    """
    return html_content

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=8080)
    # uvicorn.run(app, port=8080)

#uvicorn app_2_fio:app --host 192.168.0.75 --port 8080 --reload
#https://weevil-decent-legally.ngrok-free.app