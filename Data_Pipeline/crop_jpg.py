import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdf2image
import requests
import pandas as pd
from PIL import Image
from google.cloud import storage
from google.oauth2 import service_account


# firestore credentials
credentials = service_account.Credentials.from_service_account_file("capstone-adf24-95f58339fd01.json")
storage_client = storage.Client(credentials=credentials)
bucket = storage_client.bucket('capstone-adf24.appspot.com')

def plot(image,cmap=None):
    plt.figure(figsize=(15,15))
    plt.imshow(image,cmap=cmap) 
    
    
def detect_box(image,line_min_width=15):
    '''
    Helper function for crop_image method
    '''
    gray_scale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    th1,img_bin = cv2.threshold(gray_scale,150,225,cv2.THRESH_BINARY)
    kernal6h = np.ones((1,line_min_width), np.uint8)
    kernal6v = np.ones((line_min_width,1), np.uint8)
    img_bin_h = cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal6h)
    img_bin_v = cv2.morphologyEx(~img_bin, cv2.MORPH_OPEN, kernal6v)
    img_bin_final=img_bin_h|img_bin_v
    final_kernel = np.ones((3,3), np.uint8)
    img_bin_final=cv2.dilate(img_bin_final,final_kernel,iterations=1)
    ret, labels, stats,centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
    return stats,labels


def crop_image(path):
    '''
    Takes in path of jpg and returns cropped jpg
    '''
    image=cv2.imread(path)
    i = 2
    image = image[i:image.shape[0]-i, i:image.shape[1]-i]
    stats,labels=detect_box(image)
    min_x, min_y, max_x, max_y = image.shape[1], image.shape[0], 0, 0
    for s in stats:
        if s[0] < min_x:
            if s[0] > 30:
                min_x = s[0]
        if s[1] < min_y:
            if s[1] > 30:
                min_y = s[1]
        if s[0]+s[2] > max_x:
            if s[0]+s[2] < image.shape[1]-30:
                max_x = s[0]+s[2]
        if s[1]+s[3] > max_y:
            if s[1]+s[3] < image.shape[0]-30:
                max_y = s[1]+s[3]

    padding_x = 0.03
    padding_y = 0.1
    DEFAULT_AXIS = {
        'min_x': int(padding_x * image.shape[1]),
        'min_y': int(padding_y * image.shape[0]),
        'max_x': int(image.shape[1]-padding_x * image.shape[1]),
        'max_y': int(image.shape[0]-padding_y * image.shape[0])
    }

    # if detected area is too small
    if (max_x-min_x) * (max_y-min_y) < 0.2 * image.shape[0] * image.shape[1]: 
        min_x, min_y, max_x, max_y = DEFAULT_AXIS['min_x'], DEFAULT_AXIS['min_y'], DEFAULT_AXIS['max_x'], DEFAULT_AXIS['max_y']

    # if detected area is too big
    if (max_x-min_x) * (max_y-min_y) > 0.9 * image.shape[0] * image.shape[1]: 
        min_x, min_y, max_x, max_y = DEFAULT_AXIS['min_x'], DEFAULT_AXIS['min_y'], DEFAULT_AXIS['max_x'], DEFAULT_AXIS['max_y']

    # cv2.rectangle(image, (min_x,min_y), (max_x,max_y), (0,255,0), 2)
    # plot(image)
    crop_img = image[min_y:max_y, min_x:max_x]
    # plot(crop_img)
    return crop_img


def crop_images():
    # create a "cropped_pages" directory if not exists
    isExist = os.path.exists("cropped_pages")
    if not isExist:
       os.makedirs("cropped_pages")

    for filename in os.listdir('pages'):
        path = os.path.join(os.path.abspath(os.getcwd()) + f'\pages\{filename}')
        cropped = crop_image(path)
        # plot(cropped)
        im = Image.fromarray(cropped)

        # save cropped image to cropped_pages folder with the same filename
        save_path = os.path.join(os.path.abspath(os.getcwd()) + f'\cropped_pages\{filename}')
        im.save(save_path)

        # save jpg into firestore
        blob = bucket.blob(f'\cropped_pages\{filename}')
        blob.upload_from_filename(f'\cropped_pages\{filename}')