import requests
import argparse

#import sys
import json
import time

#sys.setdefaultencoding('UTF8')

def retrieve_reddit_data_json(subreddit_url,
                              max_out_records=5,
                              min_out_records=5
                              ):

    # Max number of items to return per reddit input file
    max_out_records = 5
    # Minimum number of items to return
    min_out_records = 5

    # Create json request
    json_data_url = subreddit_url + '/.json?limit=' + str(max_out_records)

    print(json_data_url)

    # send get request and save response as response object
    r = requests.get(url=json_data_url)

    if r.status_code == 429:
        time.sleep(int(req.headers['Retry-After']))
        r = requests.get(url=json_data_url)

    if r:

        print(r.json())

        # Seperate children from json data (i.e., information from
        # individual posts)
        children = r.json()['children']

        # Extract data from all children
        post_data = []
        for child in children:
            post_data.append(child['data'])

        return post_data


        # extracting data in json format
        #subreddit_data = r.json()['data']

    else:
        print(r.status_code)
        print("Error retrieving data from reddit server. Try again later")


def scrape_images(subreddit_json_data):

    # Search for url in output json
    subreddit_json_data['url']
    return subreddit_json_data['url']


def scrape_comments():
    pass



if __name__ == '__main__':

    reddit_url = "https://www.reddit.com"

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str,
                        default='/mnt/Data/machineLearningData/',
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

    json_data = retrieve_reddit_data_json(subreddit_url)

    print(json_data)

    url = scrape_images(json_data)

    print(url)






