import argparse
import contextlib
#import csv
import urllib
import requests
import json
from tqdm import tqdm
from bs4 import BeautifulSoup
import os
import sys
import time

#from PIL import Image, ImageFile
import cv2

fields = [
    "filename",
    "pix_height",
    "pix_width",
    "num_channels",
    "file_format",
    # Tags for each image, starting with
    # race of character present
    "human_flag",
    "dwarf_flag",
    "elf_flag",
    "halfling_flag",
    "dragonborn_flag",
    "gnome_flag",
    "halforc_flag",
    "tiefling_flag",
    "aasimar_flag",
    # Tags for classes
    "barbarian_flag",
    "bard_flag",
    "cleric_flag",
    "druid_flag",
    "fighter_flag",
    "monk_flag",
    "paladin_flag",
    "ranger_flag",
    "rogue_flag",
    "sorcerer_flag",
    "warlock_flag",
    "wizard_flag",
    # Tags for gender
    "male_flag",
    "female_flag"
]

def scrape_images(datapath, redownload_images=False,
                  scrape_all=True, num_images_to_scrape=20):
    booru_url = "https://grognard.booru.org/index.php?page=post&s=list&tags=fantasy"

    # Iterate through web pages to get all image urls and metadata
    pid = 0
    img_download_cnt = 0

    # Name of csv file to hold metadata
#    metadata_file = os.path.join(datapath, "booru.csv")
#    with open(metadata_file, 'w+') as f:
#        write = csv.writer(f)
#        fields = ["image_url", "title_info"]
#        write.writerow(fields)

    # If we are going to be redownloading images, we will need
    # to prepare a new json to hold metadata
    # Remove current one from disk
    metadata_savefile_path = os.path.join(datapath, "metadata.json")
    if os.path.exists(metadata_savefile_path) and redownload_images:
        os.remove(metadata_savefile_path)

    # Initialize dict to hold metadata and labels for image
    metadata = {}

    continue_downloading = True
    while continue_downloading:

        page = requests.get(booru_url + "&pid=" + str(pid))

        while page.status_code == 429:
            time.sleep(2.0)
            page = requests.get(booru_url + "&pid=" + str(pid))

        if page:
            soup = BeautifulSoup(page.content, 'html.parser')
            table = soup.find('div', attrs={'id':'content'})

            # If there are no thumbnails in the table, end loop. We've
            # collected all the image urls
            thumb_rows = table.findAll('span', attrs={'class': 'thumb'})
            if not thumb_rows:
                break

            for row in thumb_rows:

                image_url = row.a.img['src']
                print("downloading: ", image_url)

                # site crawl-delay is 60s, so wait that long before
                # retrieving another url

                # download image
                if download_image(image_url, datapath, redownload_images) == 1:
                    img_download_cnt += 1

                # if we have downloaded the number of images that we
                # want, stop download loop
                if scrape_all == False and img_download_cnt == num_images_to_scrape:
                    continue_downloading = False
                    break

                # save metadata to csv
#                metadata = [image_url, row.a.img['title']]

#                # Since we're going to be time limited by the crawl
#                # delay anyway, repeated disk write ops shouldn't have
#                # too much of an affect on program run time
#                with open(metadata_file, 'w') as f:
#                    write = csv.writer(f)
#                    write.writerow(metadata)

            if continue_downloading == False:
                break

            time.sleep(60)
            pid += 20

    # Save collected metadata to json file


def download_image(image_url, datapath, redownload_images=False):
    """
    Download image at given url and add to database with image metadata
    """

    filename = image_url.split('/')[-1]
    savepath = os.path.join(datapath, filename)

    # Check if the file is already saved to disk. If it is,
    # there is no need to download it again
    if os.path.isfile(savepath) and not redownload_images:
        print(filename, " already downloaded")
        return -1

    # Keep track of download attempts. If

    with open(savepath, 'wb') as outfile:

        req = urllib.request.Request(image_url, headers={'User-Agent' : 'Mozilla/5.0'})

        result = urllib.request.urlopen(req)
        filesize = int(result.headers['content-length'])
#        meta = req.info()
#        filesize = meta.getheaders("Content-Length")[0]
#        filesize = int(req.head)
        with contextlib.closing(urllib.request.urlopen(req)) as fp, tqdm(
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
                total=filesize,
                file=sys.stdout,
                desc=filename
                ) as progress:

            block_size = 1024 * 8
            while True:
                block = fp.read(block_size)
                if not block:
                    break
                datasize = outfile.write(block)
                # Update progress bar
                progress.update(datasize)

    # Save image metadata
    print("Getting metadata for image file: ", savepath)
    image = cv2.imread(savepath)
    height, width, num_channels = image.shape
    save_metadata(datapath, filename, width, height, num_channels)

    return 1


#    p = ImageFile.Parser()
#    block_size = 1024
#    while True:
#        with open(savepath, 'wb') as outfile:
#            response = requests.get(image_url, stream=True)
#            filesize = int(response.headers.get('content-length', 0))
#            with tqdm(
#                      unit="iB",
#                      unit_scale=True,
#                      unit_divisor=1024,
#                      total=filesize,
#                      file=sys.stdout,
#                      desc=filename
#                      ) as progress:
#
#                for data in response.iter_content(block_size):
#                    progress.update(len(data))
#                    outfile.write(data)
#
#        # Check that the image was downloaded correctly (not corrupted)
#        # if not, wait 60s and restart download
#        try:
#            with Image.open(savepath, mode='r') as image:
#                image.verify()
#
#                # If the image is verified succesfully, save metadata
#                img_width, img_height = image.size
#                num_channels = len(image.split())
#                save_metadata(datapath, filename, img_width,
#                              img_height, num_channels)
#                return 1
#
#        except Exception as e:
#            # Wait 60s and retry download
#            print("Download of ", filename, " failed, waiting 60s to retry")
#            time.sleep(60)

#                p.feed(data)

#                if p.image:
#                    img_width, img_height = p.image.size
#                    num_channels = len(p.image.split())
#                    save_metadata(datapath, filename, img_height,
#                                  img_width, num_channels)
#                    p.image.save(savepath)

#            save_metadata(datapath, filename)
#            return 1

#    except Exception as e:
#        print(filename, " could not be downloaded because of ", e)
#        return -1


#   except Exception as e:
#        print("error downloading: ", image_url, "due to ", e)
#        return -1

def save_metadata(datapath, filename, img_width,
                  img_height, num_channels):
    """
    Save metadata in json format
    """
    filepath = os.path.join(datapath, filename)

    # Open file to get width, height, and channels of image
#    image = Image.open(filepath, mode='r')
#    img_width, img_height = image.size
#    num_channels = len(image.split())

    # Get image type
    if filename.endswith('.png'):
        img_type = 'png'

    else:
        img_type = 'jpg'


    # Initialize dict for metadata. Note that all tags are set to '0' initially
    metadata = {
        'img_height'   : img_height,
        'img_width'    : img_width,
        'num_channels' : num_channels,
        'file_format'  : img_type,
        'tags' : {
            # DnD Races
            'Human' :     0,
            'Dwarf' :     0,
            'Elf' :       0,
            'Halfling':   0,
            'Dragonborn': 0,
            'Gnome' :     0,
            'Half-Orc':   0,
            'Tiefling':   0,
            'Aasimar':    0,
            # DnD Classes
            'Barbarian' : 0,
            'Bard' :      0,
            'Cleric':     0,
            'Druid':      0,
            'Fighter':    0,
            'Monk':       0,
            'Paladin':    0,
            'Ranger':     0,
            'Rogue':      0,
            'Sorcerer':   0,
            'Warlock':    0,
            'Wizard':     0,
            # Gender
            'Male':       0,
            'Female':     0
        }
    }

    metadata_savefile_path = os.path.join(datapath, "metadata.json")

    if not os.path.exists(metadata_savefile_path):
        with open(metadata_savefile_path, 'w') as json_file:
            json.dump({filename: metadata}, json_file, indent=4)

    else:
        with open(metadata_savefile_path, 'r+') as json_file:
            data = json.load(json_file)
            data[filename] = metadata

        with open(metadata_savefile_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)


#    metadata = [
#        filename,
#        img_height,
#        img_width,
#        num_channels,
#        img_type,
#        0,
#        0,
#        0,
#        0,
#        0,
#        0,
#        0,
#        0,
#        0,
#        0,
#        0,
#        0,
#        0,
#        0,
#        0,
#        0,
#        0,
#        0,
#        0,
#        0,
#        0,
#        0,
#        0
#    ]

#    datafile = os.path.join(datapath, "metadata.csv")
#
#    with open(datafile, 'a+') as csv_file:
#        write = csv.writer(csv_file)
#
#        # If the file doesn't exist, write in the fields
#        if not os.path.isfile(datafile):
#            write.writerow(fields)
#
#        write.writerow(metadata)

#    datafile = os.path.join(datapath, "metadata.json")
#    with open(datafile, 'w+') as json_file:
#        if overwrite_json:
#            json.dump({filename : metadata}, json_file, indent=4)
#
#        else:
#            try:
#                if os.path.isfile(datafile):
#                    buf = json_file.read()
#                    print("json buf: ", buf)
#                    data = json.loads(buf)
#                    data[filename] = metadata
#                    json.dump(data, json_file, indent=4)
#
#                else:
#                    json.dump({filename : metadata}, json_file, indent=4)
#
#
#            except Exception as e:
#                print("Json not loaded because of: ", e)
#                json.dump({filename : metadata}, json_file, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str,
                        default=r'/media/alphagoat/Backup Plus/MachineLearningData',
                        help="Directory to save character images from booru"
                        )

    parser.add_argument('--scrape_all', action='store_true',
                        default=False,
                        help="Specify to scrape all images"
                        )

    parser.add_argument('--num_images_to_scrape', type=int,
                        default=20,
                        help="Number of images to scrape from booru"
                        )

    parser.add_argument('--redownload_images', action='store_true',
                        default=False,
                        help=""""
                        Flag to raise to allow download of images that have
                        already been downloaded
                        """
                        )

    flags, _ = parser.parse_known_args()

    datapath = os.path.join(flags.dataset_path, "BooruCharacterPortraits")

    # If the datapath directory does not exist, create it
    if not os.path.isdir(datapath):
        os.mkdir(datapath)

    scrape_all_flag = flags.scrape_all

    num_images_to_scrape = flags.num_images_to_scrape

    redownload_images = flags.redownload_images

    scrape_images(datapath, redownload_images=redownload_images,
                  scrape_all=scrape_all_flag,
                  num_images_to_scrape=num_images_to_scrape)
