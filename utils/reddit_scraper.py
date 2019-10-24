import requests
import urllib
import contextlib
import argparse

#import sys
import json
import time

#sys.setdefaultencoding('UTF8')

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

    print(json_data_url)

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

    # Search for image url in output json
    image_url = post_json_data['url']
    print("image_url: ", image_url)

    if image_url.endswith('.jpg') or image_url.endswith('.png'):
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

        return image_url


def scrape_comments():
    pass



if __name__ == '__main__':

    reddit_url = "https://www.reddit.com"

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str,
                        default='/mnt/Data/machineLearningData/The_Pack_Images',
                        help='Directory to deposit scraper data'
                        )

    parser.add_argument('--data_name', type=str,
                        default='the_pack_images',
                        help='Name of dataset to be scraped'
                        )

    parser.add_argument('--subreddit', type=str,
                        default='the_pack',
                        help='Subreddit to perform scraping operation on'
                       )

    #parser.add_argument('--client_id', type=

    flags, _ = parser.parse_known_args()

    subreddit_url = reddit_url + '/r/' + flags.subreddit

    json_data_list, after_token = retrieve_reddit_data_json(subreddit_url)

    for post_data in json_data_list:
        scrape_images(post_data, flags.dataset_path)

    while after_token:

        print("after_token: ", after_token)

        json_data_list, after_token = retrieve_reddit_data_json(subreddit_url,
                                                                after_token=after_token)
        scrape_images(post_data, flags.dataset_path)

        for post_data in json_data_list:
            scrape_images(post_data, flags.dataset_path)









