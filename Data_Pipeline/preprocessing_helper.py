import PyPDF2
import pandas as pd
import requests
import io
import string
import json
import re
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import PDFPageAggregator
from pdfminer3.converter import TextConverter

def sentenceCleaning(df):
    list_of_sentences = []
    for i in range(len(df)):
        ls = df.iloc[i]['Processed_Sentences'].split("\', \' ")
        ls[0] = ls[0].replace("['","")
        ls[-1] = ls[-1].replace("']","")
        list_of_sentences.append(ls)
    df['Processed_Sentences'] = list_of_sentences
    return df

def cutLongSentences(listOfSentences, num=400):
    result = []
    for sentence in listOfSentences:
        if len(sentence) < 400:
            result.append(sentence)
    return result

def remove_duplicated_sentences(ls):
    return list(set(ls))

def doubleFilter(sentences, keywordList1, keywordList2):
    filtered_sentences = []
    for sentence in sentences:
        for keyword1 in keywordList1:
            if keyword1 in sentence.lower():
                for keyword2 in keywordList2:
                    if keyword2 in sentence.lower():
                        filtered_sentences.append(sentence)
                        break
    return filtered_sentences


def singleFilter(sentences, keywordList):
    filtered_sentences = []
    for sentence in sentences:
        for keyword in keywordList:
            if keyword in sentence.lower():
                filtered_sentences.append(sentence)
                break
    return filtered_sentences


def extract_pdf(file, verbose=False):
    """
    Process raw PDF text to structured and processed PDF text to be worked on in Python.

    Parameters
    ----------
    file : textfile
        Textfile that contains raw PDF text.

    Return
    ------
    text : str
        processed PDF text if no error is throw

    """  
    
    if verbose:
        print('Processing {}'.format(file))

    try:
        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        codec = 'utf-8'
        laparams = LAParams()

        converter = TextConverter(resource_manager, fake_file_handle, codec=codec, laparams=laparams)
        page_interpreter = PDFPageInterpreter(resource_manager, converter)
        
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()

        content = []

        for page in PDFPage.get_pages(file,
                                      pagenos, 
                                      maxpages=maxpages,
                                      password=password,
                                      caching=True,
                                      check_extractable=False):

            page_interpreter.process_page(page)

            content.append(fake_file_handle.getvalue())

            fake_file_handle.truncate(0)
            fake_file_handle.seek(0)        

        text = '##PAGE_BREAK##'.join(content)

        # close open handles
        converter.close()
        fake_file_handle.close()
        
        return text

    except Exception as e:
        print(e)

        # close open handles
        converter.close()
        fake_file_handle.close()

        return ""

def extract_content(url):
    """
    Downloads PDF text content from a given URL and parse PDF to obtain processed text.

    Parameters
    ----------
    url : str
        String that contains url to desired PDF

    Return
    ------
    text : str
        processed PDF text if no error is throw

    """   
    headers={"User-Agent":"Mozilla/5.0"}

    try:
        # retrieve PDF binary stream
        r = requests.get(url, allow_redirects=True, headers=headers)
        
        # access pdf content
        text = extract_pdf(io.BytesIO(r.content))

        # return concatenated content
        return text

    except:
        return ""


def preprocess_lines(line_input):
    """
    Helper Function to preprocess and clean sentences from raw PDF text 

    Parameters
    ----------
    line_input : str
        String that contains a sentence to be cleaned

    Return
    ------
    line : str
        Cleaned sentence

    """      
    # removing header number
    line = re.sub(r'^\s?\d+(.*)$', r'\1', line_input)
    # removing trailing spaces
    line = line.strip()
    # words may be split between lines, ensure we link them back together
    line = re.sub(r'\s?-\s?', '-', line)
    # remove space prior to punctuation
    line = re.sub(r'\s?([,:;\.])', r'\1', line)
    # ESG contains a lot of figures that are not relevant to grammatical structure
    line = re.sub(r'\d{5,}', r' ', line)
    # remove emails
    line = re.sub(r'\S*@\S*\s?', '', line)
    # remove mentions of URLs
    line = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', r' ', line)
    # remove multiple spaces
    line = re.sub(r'\s+', ' ', line)
    # join next line with space
    line = re.sub(r' \n', ' ', line)
    line = re.sub(r'.\n', '. ', line)
    line = re.sub(r'\x0c', ' ', line)
    line = line.replace(".", ".<stop>")
    line = line.replace("•", "<stop>")
    line = line.replace("?", "?<stop>")
    line = line.replace("!", "!<stop>")
    sentences = line.split("<stop>")
    sentences = sentences[:-1]
    
    return sentences


def get_processed_sentences_from_url(url):
    extracted_text = extract_content(url)
    preprocessed_sentences = preprocess_lines(extracted_text)
    processed_sentences = cutLongSentences(preprocessed_sentences)
    return processed_sentences


def generate_keyword_json():
    question_keywords = {}
    qn_name = 'Q3'
    kw_list1 = ['verified', 'verify', 'verification', 'estimates', 'certification']
    kw_list2 = ['scope 1', 'scope 2', 'scope 3', 'emissions', 'GHG']
    kw_lists = [kw_list1, kw_list2]
    question_keywords[qn_name] = kw_lists
    qn_name = 'Q4'
    kw_list1 = ["biodiversity", " nature ", " species ", " habitats ", " wildlife ", " ecosystems ", "green spaces", "greenery"]
    kw_list2 = ["support", "promote", "conserve", "preserve", "restore", "commitment", "enhance", "address", "reduce", "mitigate", "manage"]
    kw_lists = [kw_list1, kw_list2]
    question_keywords[qn_name] = kw_lists
    qn_name = 'Q7'
    kw_list1 = ["net zero", "net-zero", "carbon neutral", "carbon-neutral", "carbon neutrality", "carbon-neutrality", "climate neutral", "climate-neutral"]
    kw_list2 = ["2050", "long term", "long-term", "longer term", "20 years", "30 years"]
    kw_lists = [kw_list1, kw_list2]
    question_keywords[qn_name] = kw_lists
    qn_name = 'Q8'
    kw_list1 = ["low carbon", "low-carbon", "decarbonization", "decarbonisation", "carbon"]
    kw_lists = [kw_list1, kw_list2]
    question_keywords[qn_name] = kw_lists
    qn_name = 'Q9'
    kw_list1 = ["incentive", "remuneration", "bonus"]
    kw_list2 = ["decarbonisation", "decarbonization", "climate", "carbon", "emission"]
    kw_lists = [kw_list1, kw_list2]
    question_keywords[qn_name] = kw_lists
    qn_name = 'Q10'
    kw_list1 = ["policy", "policies", "legislation", "policy making", "policy-making", "policymakers", "trade association", "trade organization", "trade organisation", "government", "authorities", "regulatory"]
    kw_list2 = ["carbon", "climate", "net zero"]
    kw_list3 = ["engage", "support", "fund", "invest", "research", "input"]
    kw_lists = [kw_list1, kw_list2, kw_list3]
    question_keywords[qn_name] = kw_lists
    qn_name = 'Q11'
    kw_list1 = ['renewable', 'aternative source', 'solar', 'wind', 'hydroelectric', 'geothermal', 'thermal']
    kw_list2 = ['aim', 'target', 'goal', 'objective', 'obtain']
    kw_lists = [kw_list1, kw_list2]
    question_keywords[qn_name] = kw_lists
    qn_name = 'Q12'
    kw_list1 = ['low carbon', 'alternate materials', 'sustainable', 'carbon neutral']
    kw_list2 = ['supplier', 'distributor', 'provider', 'partner', 'contract', 'partnership', 'contract', 'material']
    kw_lists = [kw_list1, kw_list2]
    question_keywords[qn_name] = kw_lists
    qn_name = 'Q13'
    kw_list1 = ['carbon capture', 'hydrogen generation', 'battery storage']
    kw_list2 = ['plan', 'target', 'initiative', 'strategies','investing', 'pursuing']
    kw_lists = [kw_list1, kw_list2]
    question_keywords[qn_name] = kw_lists
    qn_name = 'Q14'
    kw_list1 = ['coal', 'coke', 'carbon', 'CO2', 'decarbonization', 'decarbonisation', 'decarbonize', 'decarbonise', 'climate', 'environment']
    kw_list2 = ['value chain']
    kw_lists = [kw_list1, kw_list2]
    question_keywords[qn_name] = kw_lists
    with open("question_keywords.json", "w") as outfile:
        json.dump(question_keywords, outfile)