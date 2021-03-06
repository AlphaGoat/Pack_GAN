import argparse
import contextlib
import datetime
import os
import requests
import sqlite3
import sys
import threading
from tqdm.auto import tqdm
import time
import urllib

from PIL import Image

class TqdmUpTo(tqdm):
    """Provides update_to(n), which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b : Blocks transferred so far
        bsize : size of each block
        tsize : total size
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n) # will also set self.n = b * bsize

def retrieve_reddit_data_json(subreddit_url,
                              limit=5,
                              after_token=None,
                              min_out_records=5
                              ):

    # Max number of items to return per reddit input file
    max_out_records = 5

    # Minimum number of items to return
    min_out_records = 5

    # Create json request
    json_data_url = subreddit_url + '/.json?limit=' + str(limit)

    if after_token:
        count = limit
        json_data_url = json_data_url + '&after=' + after_token + '&count=' + str(count)


    # send get request and save response as response object
    r = requests.get(url=json_data_url)

    while r.status_code == 429:
        time.sleep(2.0)
        r = requests.get(url=json_data_url)

    if r:
        # Seperate children from json data (i.e., information from
        # individual posts)
        children = r.json()['data']['children']

        # Extract data from all children
        post_data = []
        for child in children:
            post_data.append(child['data'])

        # Retrieve token returned for 'after' parameter, to be
        # used in next call of api
        try:
            after_token = r.json()['data']['after']
        except KeyError:
            print("unable to retrieve after_token")
            after_token = None

        # Define 'count' parameter (same as limit)
        count = limit

        return post_data, after_token


        # extracting data in json format
        #subreddit_data = r.json()['data']

    else:
        print(r.status_code)
        print("Error retrieving data from reddit server. Try again later")

        return


def scrape_images(post_json_data, data_path):
    """
    Pull image from given url and save data about image
    in json
    """
    # initialize dict to save post and image information to json
    data_json = {
                 "info":{},
                 "image": {},
                }

    # Retrieve image url
    image_url = post_json_data['url']

    if image_url.endswith('.jpg') or image_url.endswith('jpeg') or image_url.endswith('.png'):
        print("downloading: ", image_url)
        filename = image_url.split('/')[-1]
        save_path = os.path.join(data_path, filename)

        try:
            with open(save_path, 'wb') as out_file:
                filesize = int(requests.head(image_url).headers["Content-Length"])
                with contextlib.closing(urllib.request.urlopen(image_url)) as fp, tqdm(
                         unit="B",
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
                        datasize = out_file.write(block)
                        # Update progress bar
                        progress.update(datasize)

            # Get dimensions of downloaded image and save in data json
            with Image.open(save_path) as img:
                width, height = img.size

        except Exception as e:
            print("error downloading: ", image_url, "due to ", e)
            return None


    else:
        return None

    # Search for other image info
    try:
        post_url = post_json_data['permalink']
    except KeyError:
        post_url = 'NULL'

    try:
        author = post_json_data['author']
    except KeyError:
        author = 'NULL'

    try:
        post_title = post_json_data['title']
    except KeyError:
        post_title = 'NULL'

    # Get the date post was made
    try:
        created_utc = post_json_data['created_utc']
        parsed_post_date = datetime.datetime.fromtimestamp(created_utc,
                                                       datetime.timezone.utc)
    except KeyError:
        pased_post_date = 'NULL'
        pass

    data_json["info"]["image_url"] = image_url
    data_json["info"]["post_url"] = post_url
    data_json["info"]["author_username"] = author
    data_json["info"]["post_title"] = post_title
    data_json["info"]["post_date"] = parsed_post_date

    data_json["image"]["width"] = width
    data_json["image"]["height"] = height
    data_json["file_name"] = save_path

    return data_json


def scrape_comments():
    pass

class Spinner(object):
    """Object for displaying spinner on command line
       for as long as we are scraping data

       author: Victor Moyseenko (stackoverflow)
       https://stackoverflow.com/questions/4995733/how-to-create-a-spinning-command-line-cursor#4995896
    """
    busy = False
    delay = 0.1

    @staticmethod
    def spinning_cursor():
        while 1:
            for cursor in '|/-\\': yield cursor

    def __init__(self, delay=None):
        self.spinner_generator = self.spinning_cursor()
        #self.stop_spinner = stop_spinner
        if delay and float(delay): self.delay = delay

    def spinner_task(self):
        while self.busy:
            sys.stdout.write(next(self.spinner_generator))
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b')
            sys.stdout.flush()

    def __enter__(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def __exit__(self, exception, value, tb):
        #self.busy = False
        time.sleep(self.delay)
        if exception is not None:
            return False


if __name__ == '__main__':

    reddit_url = "https://www.reddit.com"

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str,
#                        default='/mnt/Data/machineLearningData/',
                        default=r'/media/alphagoat/Backup Plus',
                        help='Directory to deposit scraper data'
                        )

#    parser.add_argument('--sqlite_dbpath', type=str,
#                        default='/mnt/Data/machineLearningData/',
#                        help="Path to write sqlite database to"
#                        )
#
    parser.add_argument('--subreddit', type=str,
                        default='THE_PACK',
                        help='Subreddit to perform scraping operation on'
                       )

    parser.add_argument('--num_images_to_scrape', type=int,
                        default=5,
                        help="Number of images to scrape from chosen subreddit"
                        )

    parser.add_argument('--scrape_all', action='store_true',
                        help="""
                             Arg input to specify whether to try to scrape every image off
                             a selected subreddit.
                            """
                        )


    #parser.add_argument('--client_id', type=

    flags, _ = parser.parse_known_args()

    subreddit = flags.subreddit

    data_path = os.path.join(flags.dataset_path, subreddit)

    # If a data directory for the specific subreddit we are collecting for doesn't exist,
    # make it
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    # open sqlite database for data scraped from this subreddit, or initialize
    # a new one if one for this subreddit hasn't been initialized yet
    print("sqlite database path: ", os.path.join(data_path, subreddit + '.sqlite3'), "\n")

    # NOTE: this is only a temporary solution! find out why the database can't be written
    #       to the external hard drive!
    with sqlite3.connect(os.path.join(data_path, subreddit + '.sqlite3')) as conn:
        #subreddit_db = sqlite3.connect(os.path.join(data_path, subreddit, '.sqlite3'))
        cursor = conn.cursor()

        # Try to pull subreddit table, if it exists
        try:
            cursor.execute("""
                           SELECT name FROM sqlite_master WHERE type='table' AND name='{}_metadata';
                           """.format(subreddit))

        except sqlite3.OperationalError:
            # If it doesn't exist, create it
            cursor.execute("""
                           CREATE TABLE {}_metadata(image_id INTEGER PRIMARY KEY, filename TEXT,
                           image_url TEXT, post_url TEXT, author_username TEXT, post_title TEXT,
                           post_date DATETIME image_width INTEGER, image_height INTEGER
                           """.format(subreddit)
                           )

        subreddit_url = reddit_url + '/r/' + flags.subreddit

        # initialize variable to keep track of how many images we have pulled down
        # from the subreddit thus far

        json_data_list, after_token = retrieve_reddit_data_json(subreddit_url,
                                                                limit=flags.num_images_to_scrape)

        # Use sqlite to store image and post data we'd like to capture
        if flags.scrape_all:

            img_counter = 0

            # Initialize spinner on command line to let user know that
            # process is ongoing
#            with Spinner() as s1:
#                try:

                    # So long as an after_token was returned by the reddit json,
                    # continue the loop
            while after_token:
#                        pdb.set_trace()
                json_data_list, after_token = retrieve_reddit_data_json(subreddit_url,
                                                                        limit=flags.num_images_to_scrape,
                                                                        after_token=after_token)
#                        pdb.set_trace()
                for post_data in json_data_list:

                    saved_post_data = scrape_images(post_data, data_path)
#                            pdb.set_trace()

                    # If we gathered data from the post, meaning that the post
                    # contained an actual image
                    if saved_post_data:

                        filename = saved_post_data['file_name']

                        image_url = saved_post_data["info"]["image_url"]
                        post_url = saved_post_data["info"]["post_url"]
                        author   = saved_post_data["info"]["author_username"]
                        post_title   = saved_post_data["info"]["post_title"]
                        post_date = saved_post_data["info"]["post_date"]
                        file_name = saved_post_data["file_name"]
                        image_height = saved_post_data["image"]["height"]
                        image_width  = saved_post_data["image"]["width"]

                        # Insert scraped data into sqlite table
                        # Try to pull subreddit table, if it exists
                        try:
                            cursor.execute("""
                                           SELECT name FROM sqlite_master WHERE type='table' AND name='{}_metadata';
                                           """.format(subreddit))

                        except sqlite3.OperationalError:
                            # If it doesn't exist, create it
                            cursor.execute("""
                                           CREATE TABLE {}_metadata(image_id INTEGER PRIMARY KEY, filename TEXT,
                                           image_url TEXT, post_url TEXT, author_username TEXT, post_title TEXT,
                                           post_date DATETIME image_width INTEGER, image_height INTEGER
                                           """.format(subreddit)
                                           )

#                        cursor.execute("""
#                                       INSERT INTO {}_metadata(file_name, image_url, post_url,
#                                       author_username, post_title, post_date, image_width, image_height)
#                                       VALUES(?,?,?,?,?,?,?,?)
#                                       """.format(subreddit),
#                                       (file_name, image_url, post_url, author, post_title, post_date,
#                                        image_width, image_height))

                        # commit to database before moving onto next image
                        conn.commit()

                        # Add to image counter
                        img_counter += 1

                        # Post image count for every hundreth image
                        if img_counter % 100 == 0:
                            print("Number of images scraped: {}".format(img_counter))

                # If a specified number of images to scrape was provided by the user
                # (and the argument 'scrape_all' was not used), see if we have
                # already collected that number of images. If so, break the loop
                if img_counter == flags.num_images_to_scrape and not flags.scrape_all:
                    after_token = False
                    break

#                    # Stop spinner
#                    s1.busy = False
#                    s1.join()

                # Print number of images scraped
                print("{0} images were succesfully scraped from r/{1}".format(img_counter, flags.subreddit))

#                except Exception as e:
#                    # If an exception is reached, stop spinner
#                    s1.busy = False
#                    s1.join()
#                    print("image scraping operating halted: ", e)



