"""
Stores tags assigned to images pulled from reddit in a json saved to that data directory
JSON FORMAT: (based of the COCO dataset format)
{image_id: {
             "info": info,
             "images": [image],
             "annotations": [annotation],
            }

            info{
                 "year"                  : int,
                 "post_title"            : str,
                 "contributor_username"  : str,
                 "date_posted"           : datetime,
                 "reddit_url"            : str,
                 "tags"                  : [int],
                 "image_text"            : str,
                }

            image{
                  "id"                   : int,
                  "width"                : int,
                  "height"               : int,
                  "file_name"            : str,
                }
}
"""
import os
import json

def save_json_info(data_json, info_dict):

    try:
        with open(data_json,


