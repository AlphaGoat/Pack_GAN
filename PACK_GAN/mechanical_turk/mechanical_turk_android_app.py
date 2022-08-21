import kivy
from kivy.app import App

kivy.require('1.9.0')

from kivy.uix.image import AsyncImage
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.button import Button
#from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput

#from kivy.config import Config

from kivy.network.urlrequest import UrlRequest

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.poolmanager import PoolManager

import socket

#Config.set('graphics', 'resizable', True)

class MainFrame(GridLayout):
    '''Custom float layout widget to control layout of image,
       tags, and buttons in app
    '''
    def __init__(self, **kwargs):

        # make sure we aren't overriding any important functionality
        super(MainFrame, self).__init__(**kwargs)
        self.cols = 1


        # Store the address of the image server
        #self.image_server_address = image_server_address

        ## FOR TESTING PURPOSES: just load up a single image
        self.image_path = "/home/peter/Pictures/background.png"
        self.image = AsyncImage(source=self.image_path)

        # adding image to main frame
        self.add_widget(self.image)

        # initialize grid for image tags
        self.image_tags_grid = GridLayout()

        # include 5 tags per column


        # Create grid for submission, next, and previous image buttons
        # at bottom of main frame
        self.bottom_grid = GridLayout()
        self.bottom_grid.cols = 3

        # Button to cycle to previous image
        self.prev_image_button = Button(text="Previous", font_size=40)
        self.bottom_grid.add_widget(self.prev_image_button)


        # Button to go to next image
        self.next_image_button = Button(text="Next", font_size=40)
        self.bottom_grid.add_widget(self.next_image_button)


        # Submission button
        self.submit_button = Button(text="Submit", font_size=40)
        self.bottom_grid.add_widget(self.submit_button)

        # Add the bottom grid to the main frame
        self.add_widget(self.bottom_grid)


class ServerLoginScreen(Popup):

    def __init__(self, **kwargs):

        super(ServerLoginScreen, self).__init__(**kwargs)

        # Set auto dismiss to false so user cannot automatically
        # bypass this popup screen
        #self.auto_dismiss = False

        # grid layout for popup
        self.main_frame = GridLayout()
        self.add_widget(self.main_frame)

        # Variable containing the address of the default image server
        # The user can change this if they want
        self.default_server_address = None

        # Line entry for server address
        self.server_address_entry = TextInput(text=self.default_server_address)
        self.main_frame.add_widget(self.server_address_entry)

        # Line entry for password. User must provide this themselves
        self.password_entry = TextInput(text="password")
        self.main_frame.add_widget(self.password_entry)

        # submission button. Checks to see if password is correct,
        # then dismisses popup and brings up main frame
        self.login_submit_button = Button(text="Submit")

    def check_password(self):
        """
        Bound method checking user input password. If the password
        is correct, send request to image server and bring up main
        frame
        """
        server_address = self.server_address_entry.text
        password = self.password_entry.text

        payload = {'password': password}








class MechanicalTurkApp(App):

    def build(self):

        # Some code allowing you to contact image server
        ### IMAGE SERVER CODE #####
        ##########################

        # Initial popup requesting password to image server
        self.server_login_popup = ServerLoginScreen()
        self.server_login_popup.open()

        self.main_frame = MainFrame()


        ## loading image
        #self

        ## Formatting image
        #self.img.allow_stretch = True
        #self.img.keep_ratio = False

        ## Providing size to image (varies from 0 to 1)
        #self.img.size_hint_x = 1
        #self.img.size_hint_y = 1

        ## position set
        #self.img.pos

        # Load in labels for tags



        # Buttons to submit chosen tags and load next image

class SourcePortAdapter(HTTPAdapter):
    """
    Transport adapter allowing us to set the source port.
    adopted from Martijin Pieters on stackoverflow

    https://stackoverflow.com/questions/47202790/python-requests-how-to-specify-port-for-outgoing-traffic?rq=1
    """
    def __init__(self, port, *args, **kwargs):
        self._source_port = port
        super(SourcePortAdapter, self).__init__(*args, **kwargs)

    def init_poolmanager(self, connections, maxsize, block=False):
        self.poolmanager = PoolManager(
            num_pools=connections, maxsize=maxsize,
            block=block, source_address=('', self._source_port))


class Client():
    def __init__(self,


#    def load_next_image(self):
#        #self.img = Image(source=
#        pass
#
#
#
#    def contact_image_server(self, req, result):
#        pass
#


class TagLabels(Label):
    pass


if __name__ == '__main__':
    mechanical_turk_app = MechanicalTurkApp()
    mechanical_turk_app.run()
