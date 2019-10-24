"""
Utility for manually applying tags to imagery downloaded from
R/THE_PACK

Peter J. Thomas, 23 Oct 2019
"""
from PIL

class MechanicalTurk(object):

    def __init__(self):

        pass

    def get_exif(self, filename):
        """
           Retrieve Exchangeable image file format (Exif) data from
           image. Courtesy of Jayson DeLancey.

           https://developer.here.com/blog/getting-started-with-geocoding-exif-image-metadata-in-python3
        """
        image = Image.open(filename)
        image.verify()
        return image._getexif()

    def tag_imagery(self, filename):

        pass

