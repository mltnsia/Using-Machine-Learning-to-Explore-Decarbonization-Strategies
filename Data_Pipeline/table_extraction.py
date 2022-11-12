import os
import re
import requests
import io
import pandas as pd
from PyPDF2 import PdfFileReader


units = ['tonc02 eq', 'tonnes co2e', 'thousand tonnes co₂e', 'tco2e', 'mn t co2 equivalent', 'mn t co 2equivalent', 'metric tonnes co2e',
        'metric tonnes co₂e', 'tonnes co2', 'thousand tco2e', 't co2e', 'thousand tons of carbon dioxide equivalent', 'metric tons of co2e',
        'mn t co2 eq ', 'thousand tonnes co2e ', 'thousand tco2e', 'mtco2e', 'tonco₂e', 'tonnes co₂ eq', 'tco2eq',
        'metric tons co2e', 'metric tons co₂-e(t)', 'tonnes co₂e', 'ton co2 eq', 'mtco2eq', 'mn tonnes co2 eq', 'mn tonnes co 2 eq',
        'million tonnes co2e','million metric tons co2 equivalent', 'million tonnes co₂e', 'million tonnes of со2e',
        'mmt co2 e', 'kt co₂-e', 'millions of tco2e', 'ktco2e', 'co2e tonnes', 'tonnes co2e', 'metric tons co2e', 'mt co2e', 'million tonnes co2e']
years = ["2021", "2020", "2019", "FY21", "FY20", "FY19"]
kw_list1 = ["scope 1", "scope 2", "scopes 1"]
kw_list2 = ["scope 3"]


def contains_table(text, kw_list):
    '''
    Returns True if a page contains relevant table else False

    Input
    =====
    text: str; text parsed from a page of the report
    kw_list: list of str; question-specific keywords

    Output
    ======
    boolean
    '''
    contains_unit = False
    contains_years = False
    contains_numbers = False
    contains_kw = False

    for unit in units:
        if unit in text.lower():
            contains_unit = True
            break
    
    year_count = 0
    for year in years:
        if year in text:
            year_count += 1
    contains_years = True if year_count >= 2 else False

    number_count = 0
    for word in text.split(" "):
        if re.match("^20\d\d$", word):
            continue
        else:
            word = word.replace(",", "")
            word = word.replace(".", "")
            if all(char.isdigit() for char in word):
                number_count += 1
    contains_numbers = True if number_count >= 10 else False

    for kw in kw_list:
        if kw in text.lower():
            contains_kw = True
            break

    return all([contains_unit, contains_years, contains_numbers, contains_kw])


def extract_page_numbers(df, qn):
    '''
    Updates and returns df with a column of extracted page numbers

    Input
    =====
    df: pandas.DataFrame; contains a "Company" column and "URL" column
    qn: {1, 2}; question in the decarbonization framework

    Output
    ======
    pandas.DataFrame with an additional column named "Q1" or "Q2" depending on qn
    '''
    if qn == 1:
        kw_list = kw_list1
    elif qn == 2:
        kw_list = kw_list2
    else:
        raise Exception("qn can only take values 1 or 2")

    all_retrieved_pages = []
    for i in range(len(df)):
        try:
            url = df.iloc[i]['Report URL']
            response = requests.get(url)
            file = io.BytesIO(response.content)
            pdf = PdfFileReader(file)
            
            retrieved_pages = []
            retrieved_page_obj = []
            curr_page = 0
            for page in pdf.pages:
                text = page.extract_text()
                text = text.replace("\n", "")
                if contains_table(text, kw_list):
                    retrieved_pages.append(curr_page+1)
                    retrieved_page_obj.append(page)
                curr_page += 1

            # if > 1 pages found, check number of digits and take max
            if len(retrieved_pages) > 1:
                retrieved_pages_new = []
                number_count_dict = {}
                for i in range(len(retrieved_pages)):
                    page = retrieved_page_obj[i]
                    text = page.extract_text()
                    number_count = len(re.findall("\d", text))
                    number_count_dict[retrieved_pages[i]] = number_count

                max_count = max(number_count_dict.values())
                for k in number_count_dict:
                    if number_count_dict[k] == max_count:
                        retrieved_pages_new.append(k)
                retrieved_pages = retrieved_pages_new
            all_retrieved_pages.append(retrieved_pages)

        except Exception as e:
            all_retrieved_pages.append("Error")
            continue

    df[f'Q{qn}'] = all_retrieved_pages
    return df