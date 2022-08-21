from django.db import models
from django.utils import timezone
from django.core.files.storage import FileSystemStorage

import datetime
#import os

image_storage = FileSystemStorage(location='/mnt/Data/machineLearningData/The_Pack_Images')

# Create your models here.
class Tag(models.Model):
    tag_text = models.CharField(max_length=200)
    # TODO: Add in functionality to count the number of times
    #       a tag is associated with an image

    def __str__(self):
        return self.tag_text

class Image(models.Model):

    # Make the image path the primary key (i.e., what each image
    # instance is referred to internally on the webserver)
    image_name = models.CharField(max_length=200, primary_key=True)
    data_path = models.CharField(max_length=200)
    pub_date = models.DateTimeField('date published')

    photo = models.ImageField(storage=image_storage)

    # Possible field to be added later, if I can modify my scraping
    # script to retrieve the post title of the reddit submission
    # associated with the image.
    # post_title = models.CharField(max_length=200)
    # post_date = models.DateTimeField('date posted to reddit')

    tags = models.ManyToManyField(Tag)

    def __str__(self):
        return self.image_name

    def was_published_recently(self):
        now = timezone.now()
        return now - datetime.timedelta(days=1) <= self.pub_date <= now



#def retreive_possible_tags(tag_file):
#    """Retreives tags that can be used to label imagery"""
#    with open(tag_file) as

