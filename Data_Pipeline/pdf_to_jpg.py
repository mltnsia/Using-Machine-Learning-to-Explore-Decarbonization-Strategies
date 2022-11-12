import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdf2image
import requests
import pandas as pd
import re
from PIL import Image
from google.cloud import storage
from google.oauth2 import service_account


# firestore credentials
credentials = service_account.Credentials.from_service_account_file("capstone-adf24-95f58339fd01.json")
storage_client = storage.Client(credentials=credentials)
bucket = storage_client.bucket('capstone-adf24.appspot.com')

def convert_to_jpg_qn_1(df):
    # remove empty list
    df = df[df["Q1"].str.contains("\[]") == False]

    # remove Error
    df = df[df["Q1"].str.contains("Error") == False]

    # convert from string to list
    df["Q1"] = df["Q1"].apply(lambda x: x[1:-1].split(","))

    # remove "/" 
    df['IssuerName'] = df['IssuerName'].str.replace("/", "")

    # remove blank space
    df['IssuerName'] = df['IssuerName'].str.replace(" ", "")
    
    # remove trailing "."
    df["IssuerName"] = df["IssuerName"].apply(lambda x: x[:-1] if (x[-1] == ".") else x)

    for index, row in df.iterrows():
        # create a "pages" directory if not exists
        isExist = os.path.exists("pages")
        if not isExist:
           os.makedirs("pages")
        try:
            pdf = requests.get(row["Report URL"])
            pages = pdf2image.convert_from_bytes(pdf.content)
            for num in row["Q1"]:
                path = 'pages/1_' + row["IssuerName"] + '.jpg'

                # save jpg into local directory
                pages[int(num)].save(path, "JPEG")

                # save jpg into firestore
                blob = bucket.blob(path)
                blob.upload_from_filename(path)

        except Exception as e:
            print(row["IssuerName"], ": FAIL")
            print(e)
                
                
def convert_to_jpg_qn_2(df):
    # remove empty list
    df = df[df["Q2"].str.contains("\[]") == False]
    
    # remove Error
    df = df[df["Q2"].str.contains("Error") == False]
    
     # convert from string to list
    df["Q2"] = df["Q2"].apply(lambda x: x[1:-1].split(","))
    
    # remove "/"
    df['IssuerName'] = df['IssuerName'].str.replace("/", "")
    
    # remove blank space
    df['IssuerName'] = df['IssuerName'].str.replace(" ", "")
    
    # remove trailing "."
    df["IssuerName"] = df["IssuerName"].apply(lambda x: x[:-1] if (x[-1] == ".") else x)

    for index, row in df.iterrows():
        # create a "pages" directory if not exists
        isExist = os.path.exists("pages")
        if not isExist:
           os.makedirs("pages")
        
        try:
            pdf = requests.get(row["Report URL"])
            pages = pdf2image.convert_from_bytes(pdf.content)
            for num in row["Q2"]:
                path = 'pages/2_' + row["IssuerName"] + '.jpg'

                # save jpg into local directory
                pages[int(num)].save(path, "JPEG")

                # save jpg into firestore 
                blob = bucket.blob(path)
                blob.upload_from_filename(path)

        except Exception as e:
            print(row["IssuerName"], ": FAIL")
            print(e)
            
