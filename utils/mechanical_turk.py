"""
Utility for manually applying tags to imagery downloaded from
R/THE_PACK

Peter J. Thomas, 23 Oct 2019
"""
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QLabel
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

from PIL import Image

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

#    def get_tags(self, exif):
#
#        if not exif:
#            raise ValueError("No EXIF metadata found")
#
#        img_tags = []
#        for (idx, tag) in TAGS.items():
#            if tag == 'ImgTag':
#                if idx not in exif:
#                    raise ValueError("No EXIF image tag found")
#

    def tag_imagery(self, filename):

        pass

class MechanicalTurk_GUI(QWidget):

    def __init__(self):

        super().__init__()

        self.title = 'R/THE_PACK Mechanical Turk'
        self.left = 10
        self.top = 10
        self.width = 320
        self.height = 60
        self.initUI()

    def initUI(self):

        self.WindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        editBox = QLineEdit('Drag this', self)
        editBox.setDragEnabled(True)
        editBox.move(10, 10)
        editBox.resize(100, 32)

        button = CustomLabel('Drop here.', self)
        button.move(130, 15)

        self.show()

    @pyqtSlot()
    def on_click(self):

        print('PyQt5 button click')

class CustomLabel(QLabel):

    def __init__(self, title, parent):

        super.__init__(title, parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat('text/plain'):
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        self.setText(e.mimeData().test())

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = MechanicalTurk_GUI()
    sys.exit(app.exec_())

