from PIL import Image
import io

class ImageGenerator():

    def __init__(self) -> None:
        self.image_path = "test_image.JPG"

    def predict(self):
        return self.image_to_byte_array(Image.open(self.image_path))


    def image_to_byte_array(self, image:Image):
        imgByteArr = io.BytesIO()
        image.save(imgByteArr, format=image.format)
        imgByteArr = imgByteArr.getvalue()
        return imgByteArr