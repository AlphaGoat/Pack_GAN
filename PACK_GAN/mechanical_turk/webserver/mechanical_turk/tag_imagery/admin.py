from django.contrib import admin

# Register your models here.
from .models import Image, Tag

admin.site.register(Image)
admin.site.register(Tag)
