from django.shortcuts import render
from django.http import HttpResponse

from .models import Image

# Create your views here.
def detail(request, image_id):
    return HttpResponse("You're looking at image %s." % image_id)

def results(request, image_id):
    """
    Web page displaying graph of tags assigned to an image as well
    as the number of times it has been assigned to the image.
    """
    return HttpResponse("You're looking at the results of image %s." % image_id)

def tag(request, image_id):
    """
    View for tagging imagery
    """
    return HttpResponse("You're tagging image %s." % image_id)

def statistics(request):
    """
    Display for statistics about dataset you would choose to display
    """
    return HttpResponse("Statistics for the r/THE_PACK dataset")

def start_page(request):
    """
    Front end for tagging utility.
    """
    return HttpResponse("Start page for imagery tagging utility.")


def index(request):
    latest_image_list = Image.objects.order_by('-pub_date')[:5]
    output = ', '.join([i.image_name for i in latest_image_list])
    return HttpResponse(output)
