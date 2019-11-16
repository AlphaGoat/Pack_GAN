from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.behaviors import DragBehavior
from kivy.uix.image import AsyncImage

import csv

class MechanicalTurkApp(App):

    def __init__(self, image_tags, data_links):

        self.image_tags = image_tags
        self.data_links = data_links
        self.img_idx = 0
        pass

    def build(self):

        # First popout frame asking user to identify themselves
        user_id_frame = GridLayout()

        main_frame = BoxLayout()
        self.image_frame = BoxLayout()
        possible_labels_frame = GridLayout()
        #chosen_labels_frame = BoxLayout()
        submission_frame = GridLayout()

        # Define buttons for submission frame
        submit_button = Button(text="SUBMIT MFER")
        submit_button.bind(on_press=self.load_next_image)


        goback_button = Button(text="PREVIOUS IMAGE")

        for tag in self.image_tags:
            label = TagLabels(text=tag)
            possible_labels_frame.add_widget(label)

    def load_next_image(self):
        '''
        Activated when user presses submit button.
        Cycles to next image in dataset
        '''
        # TODO: tie button states to the completion of this
        #       function so that pressed buttons are reset
        #       to their unpressed state

        # clear previous image from image frame
        self.image_frame.config(image='')

        self.img_idx += 1

        # load in new image from data server
        self.curr_img = AsyncImage(source=self.data_links[self.img_idx])


        # apply loaded image to image frame
        self.image_frame.add_widget(self.curr_img)


    def load_previous_image(self):
        '''
        Activated when user presses go back burron.
        Goes back to previous image and allows user
        to change labels they assigned to it
        '''
        # TODO: tie button states to the completion of this
        #       function so that pressed buttons are reset
        #       to their unpressed state

        # clear current image from frame
        self.image_frame.config(image='')

        self.img_idx -= 1

        # load in previous image from data server
        prev_img = AsyncImage(source=self.data_links[self.img_idx])

        # apply loaded img to image frame
        self.image_frame.add_widget(prev_img)

        return














class ImageWindow(Window):

    def __init__(self):

        pass



# TODO: Redefine this class as a draggable lable that can be
#       dragged onto the image
class TagLabels(ToggleButton):

    def on_press(self):
        pass





if __name__ == '__main__':

    with open('tags.txt') as tags_file:
        tags_reader = tags_file.read()

        tags_list = []
        for tag in tags_reader:
            tag = tag.strip()
            if not tag: break
            tags_list.append(tag)

    MechanicalTurkApp(tags_list).run()


