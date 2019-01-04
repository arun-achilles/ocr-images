from wand.image import Image
from PIL import Image as PI
from tesserocr import PyTessBaseAPI
import io
import os

txt = False
confidences = False

image_path = '../images'
processed_image_path = ''

image_files = os.listdir(image_path)
with PyTessBaseAPI() as api:
    for img in image_files:
        with Image(image=img) as img_page:
            with PI.open(io.BytesIO(img_page.make_blob('jpeg'))) as im:
                api.SetImage(im)
                api.Recognize()
                txt = api.GetUTF8Text()
                confidences = api.AllWordConfidences()
if confidences:
    confi_total = 0
    for i in confidences:
        confi_total = confi_total + i

    avg_confi = confi_total/len(confidences)
    print("Confidence is {0}".format(avg_confi))

def calculate_alf():
    return