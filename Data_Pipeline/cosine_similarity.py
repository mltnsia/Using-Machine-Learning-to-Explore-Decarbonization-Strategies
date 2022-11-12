import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing_helper import *

q5 = "What does your medium term (5 10 years 2025 2030) Scope 1 - 2 target (excl. carbon credits) equate to in % reduction? i.e. emissions scope, base year, target year and % reduction"
q5_kwlist1 = ['medium term', '5 years', '10 years', '2025', '2030', 'target', 'goal', 'reduction', 'decrease']
q5_kwlist2 = ['scope 1', 'scope 2', 'scopes 1']

q6 = "What is your medium term (5 10 years 2025 2030) target for reduction in Scope 3 Downstream emissions (e.g. tenant/purchaser activity) and Upstream emissions (e.g. embodied emissions in purchased construction materials)? i.e. emissions scope, base year, target year and % reduction"
q6_kwlist1 = ['medium term', '5 years', '10 years', '2025', '2030', 'target', 'goal', 'reduction', 'decrease']
q6_kwlist2 = ['scope 3']


def extract_most_relevant_sentences(question, list_of_list_of_sentences, n):
    """
    Helper function to extract n most relevant sentences using cosine similarity
    """
    vectorizer = CountVectorizer(stop_words='english')
    question_vector = vectorizer.fit_transform([question])
    best_sentences = []
    best_angles = []

    for ls in list_of_list_of_sentences:
        all_sentences = []
        all_angles = []
        for s in ls:
            # calculate cosine similarity for each sentence
            sentence_vector = vectorizer.transform([s])
            cos = cosine_similarity(question_vector, sentence_vector)
            radian = np.arccos(cos)
            angle = np.rad2deg(radian)

            # additional filtering - must contain (kw1+kw2) or (kw1+kw2+"20")
            kw_list1 = ["%", "percent", "net zero", "net-zero", "carbon neutral", "carbon neutrality", "mtco2e", "tons", "tonnes"]
            kw_list2 = ["by 20", "as early as 20", "for 20", "by the end of 20", "by year-end 20", "by year end 20"]
            kw_list3 = ["target", "goal"]
            appended = False
            for kw1 in kw_list1:
                if kw1 in s.lower():
                    for kw2 in kw_list2:
                        if kw2 in s.lower():
                            all_sentences.append(s)
                            all_angles.append(angle[0][0])
                            appended = True
                            break
                    if not appended:
                        for kw3 in kw_list3:
                            if kw3 in s.lower() and "20" in s:
                                all_sentences.append(s)
                                all_angles.append(angle[0][0])
                                appended = True
                                break
                if appended:
                    break
        
        # get top n most relevant sentences and concat them together
        temp = pd.DataFrame({'sentence': all_sentences, 'angle': all_angles})
        temp = temp.sort_values(by='angle').head(n)
        best_sentences.append("\n".join(temp['sentence']))
        best_angles.append(temp['angle'])
            
    return best_sentences, best_angles


def postprocess_sentence(text):
    '''
    Helper function to format extracted sentences
    '''
    if text != None:
        # process awkward formatting of decimals, e.g. 1.', '5 million -> 1.5 million
        text = text.replace(".', '", ".")
        # remove symbol formatting
        text = re.sub("\W*uf...", " ", text)
        # remove page break indicators
        text = text.replace("##PAGE_BREAK##", "")
    return text


def extract_relevant_sentences(df, qn):
    '''
    Main function to call for Q5 and Q6

    Input
    =====
    df: pandas.DataFrame; Contains "Company", "URL", and "Processed_Sentences" columns
    qn" {5, 6}; question in the decarbonization framework

    Output
    ======
    pandas.DataFrame with an additional column named "Q5" or "Q6" depending on qn
    '''
    if qn == 5:
        question = q5
        keywordList1, keywordList2 = q5_kwlist1, q5_kwlist2
    elif qn == 6:
        question = q6
        keywordList1, keywordList2 = q6_kwlist1, q6_kwlist2
    else:
        raise("qn should be 5 or 6")

    filtered_sentences = []
    for i in range(len(df)):
        try:
            processed_sentences = cutLongSentences(df.iloc[i]['Processed_Sentences'])
            extracted_sentences = remove_duplicated_sentences(doubleFilter(processed_sentences, keywordList1, keywordList2))
            filtered_sentences.append(extracted_sentences)
        except:
            filtered_sentences.append("")

    best_sentences, q5_best_angles = extract_most_relevant_sentences(question, filtered_sentences, 3)
    best_sentences_processed = [postprocess_sentence(s) for s in best_sentences]
    df[f'Q{qn}'] = best_sentences_processed

    return df