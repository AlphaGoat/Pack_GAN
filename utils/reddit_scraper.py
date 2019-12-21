import requests
import urllib
import contextlib
import argparse

#import sys
import os
import sqlite3
import time
import datetime

from PIL import Image

#sys.setdefaultencoding('UTF8')

# Retrieve the current image id from the number of images
# currently in the data directory


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

    # Search for image url in output json as well as other relevant info
    image_url = post_json_data['url']
    post_url = post_json_data['permalink']
    author = post_json_data['author']
    post_title = post_json_data['title']

    # Get the date post was made
    created_utc = post_json_data['created_utc']
    parsed_post_date = datetime.utcfromtimestamp(created_utc)

    data_json["info"]["image_url"] = image_url
    data_json["info"]["post_url"] = post_url
    data_json["info"]["author_username"] = author
    data_json["info"]["post_title"] = post_title
    data_json["info"]["post_date"] = post_date

    if image_url.endswith('.jpg') or image_url.endswith('jpeg') or image_url.endswith('.png'):
        filename = image_url.split('/')[-1]
        save_path = data_path + filename

        with open(save_path, 'wb') as out_file:
            with contextlib.closing(urllib.request.urlopen(image_url)) as fp:
                block_size = 1024 * 8
                while True:
                    block = fp.read(block_size)
                    if not block:
                        break
                    out_file.write(block)

        # Get dimensions of downloaded image and save in data json
        with Image.open(save_path) as img:
            width, height = img.size

        data_json["image"]["width"] = width
        data_json["image"]["height"] = height
        data_json["file_name"] = save_path

        return data_json


def scrape_comments():
    pass



if __name__ == '__main__':

    reddit_url = "https://www.reddit.com"

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str,
                        default='/mnt/Data/machineLearningData/The_Pack_Images',
                        help='Directory to deposit scraper data'
                        )

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
        os.mkdir(data_path, exist_ok=True)

    # open sqlite database for data scraped from this subreddit, or initialize
    # a new one if one for this subreddit hasn't been initialized yet
    print("sqlite database path: ", os.path.join(data_path, subreddit + '.sqlite3'))
    with sqlite3.connect(os.path.join(data_path, subreddit, '.sqlite3')) as conn:
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
            for post_data in json_data_list:

                saved_post_data = scrape_images(post_data, data_path)

                filename = saved_post_data['file_name']

                image_url = saved_post_data["info"]["image_url"]
                post_url = saved_post_data["info"]["post_url"]
                author   = saved_post_data["info"]["contributor_username"]
                post_title   = saved_post_data["info"]["post_title"]
                post_date = saved_post_data["info"]["post_date"]
                file_name = saved_post_data["image"]["file_name"]
                image_height = saved_post_data["image"]["image_height"]
                image_width  = saved_post_data["image"]["image_width"]

                # Insert scraped data into sqlite table
                cursor.execute("""
                               INSERT INTO {}_metadata(file_name, image_url, post_url,
                               author_username, post_title, post_date, image_width, image_height)
                               VALUES(?,?,?,?,?,?,?,?)
                               """, (file_name, image_url, post_url, author, post_title, post_date,
                                     image_width, image_height))

                # commit to database before moving onto next image
                conn.commit()



            while after_token:


                json_data_list, after_token = retrieve_reddit_data_json(subreddit_url,
                                                                        count=flags.num_images_to_scrape,
                                                                        after_token=after_token)

                for post_data in json_data_list:
                    saved_post_data = scrape_images(post_data, data_path)

                    filename = saved_post_data['file_name']

                    image_url = saved_post_data["info"]["image_url"]
                    post_url = saved_post_data["info"]["post_url"]
                    author   = saved_post_data["info"]["contributor_username"]
                    post_title   = saved_post_data["info"]["post_title"]
                    post_date = saved_post_data["info"]["post_date"]
                    file_name = saved_post_data["image"]["file_name"]
                    image_height = saved_post_data["image"]["image_height"]
                    image_width  = saved_post_data["image"]["image_width"]

                    # Insert scraped data into sqlite table
                    cursor.execute("""
                                   INSERT INTO {}_metadata(file_name, image_url, post_url,
                                   author_username, post_title, post_date, image_width, image_height)
                                   VALUES(?,?,?,?,?,?,?,?)
                                   """, (file_name, image_url, post_url, author, post_title, post_date,
                                         image_width, image_height))

                    # commit to database before moving onto next image
                    conn.commit()

        else:
            img_counter = 0

            for post_data in json_data_list:

                if img_counter == flags.num_images_to_scrape and flags.scrape_all != True:
                    break

                saved_post_data = scrape_images(post_data, data_path)

                filename = saved_post_data['file_name']

                image_url = saved_post_data["info"]["image_url"]
                post_url = saved_post_data["info"]["post_url"]
                author   = saved_post_data["info"]["contributor_username"]
                post_title   = saved_post_data["info"]["post_title"]
                post_date = saved_post_data["info"]["post_date"]
                file_name = saved_post_data["image"]["file_name"]
                image_height = saved_post_data["image"]["image_height"]
                image_width  = saved_post_data["image"]["image_width"]

                # Insert scraped data into sqlite table
                cursor.execute("""
                               INSERT INTO {}_metadata(file_name, image_url, post_url,
                               author_username, post_title, post_date, image_width, image_height)
                               VALUES(?,?,?,?,?,?,?,?)
                               """, (file_name, image_url, post_url, author, post_title, post_date,
                                     image_width, image_height))

                # commit to database before moving onto next image
                conn.commit()

                img_counter += 1


            while after_token:

                if img_counter == flags.num_images_to_scrape:
                    break


                json_data_list, after_token = retrieve_reddit_data_json(subreddit_url,
                                                                        count=flags.num_images_to_scrape,
                                                                        after_token=after_token)

                for post_data in json_data_list:
                    saved_post_data = scrape_images(post_data, data_path)
                    img_counter += 1

                    filename = saved_post_data['file_name']

                    image_url = saved_post_data["info"]["image_url"]
                    post_url = saved_post_data["info"]["post_url"]
                    author   = saved_post_data["info"]["contributor_username"]
                    post_title   = saved_post_data["info"]["post_title"]
                    post_date = saved_post_data["info"]["post_date"]
                    file_name = saved_post_data["image"]["file_name"]
                    image_height = saved_post_data["image"]["image_height"]
                    image_width  = saved_post_data["image"]["image_width"]

                    # Insert scraped data into sqlite table
                    cursor.execute("""
                                   INSERT INTO {}_metadata(file_name, image_url, post_url,
                                   author_username, post_title, post_date, image_width, image_height)
                                   VALUES(?,?,?,?,?,?,?,?)
                                   """, (file_name, image_url, post_url, author, post_title, post_date,
                                         image_width, image_height))

                    conn.commit()




