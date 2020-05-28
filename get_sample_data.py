import os
import requests
import io
from PIL import Image

if __name__ == '__main__':
    sample_url = 'https://safebooru.org//images/2394/33efb8875cc23f646fde9bacf20e5ce9bbe869c4.jpg?2494790'
    out_dir = 'predict_from'
    os.makedirs(out_dir, exist_ok=True)

    img_res = requests.get(sample_url, stream=True)
    img = Image.open(io.BytesIO(img_res.content))
    W, H = img.size
    x_buf = 64
    img = img.crop((x_buf, 0, W - x_buf, W - x_buf * 2))
    img = img.resize((256, 256))
    img.save(out_dir + '/sample.png')