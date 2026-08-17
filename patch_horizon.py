import easyocr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

reader = easyocr.Reader(['en'], gpu=False, verbose=False)
file = r'f:\Downloads\ieee_final_fixed\figures\fig_horizon_pe.png'
img = Image.open(file)
draw = ImageDraw.Draw(img)
results = reader.readtext(file)
for (bbox, text, prob) in results:
    text_lower = text.lower()
    if '45' in text_lower:
        print(f'{file}: Found "{text}" at {bbox}')
        x0, y0 = int(bbox[0][0]), int(bbox[0][1])
        x1, y1 = int(bbox[2][0]), int(bbox[2][1])
        draw.rectangle([x0-2, y0-2, x1+2, y1+2], fill='white')
        try:
            font = ImageFont.truetype('arial.ttf', size=int(y1-y0))
        except:
            font = ImageFont.load_default()
        draw.text((x0, y0), '1 h', fill='black', font=font)
img.save(file)
print(f'Patched {file}')
