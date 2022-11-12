from preprocessing_helper import *
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from itertools import combinations,product

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,accuracy_score,classification_report,plot_confusion_matrix,roc_curve,recall_score,confusion_matrix
import warnings
warnings.filterwarnings("ignore")

def sentenceCleaning(df):
    list_of_sentences = []
    for i in range(len(df)):
        ls = df.iloc[i]['Processed_Sentences'].split("\', \' ")
        ls[0] = ls[0].replace("['","")
        ls[-1] = ls[-1].replace("']","")
        list_of_sentences.append(ls)
    df['Processed_Sentences'] = list_of_sentences
    return df

def getMergedSentences(qn_name, kw_lists, df3):
    # Filter for relevant sentences
    sentences_qn = []
    for i in range(len(df3)):
        if len(kw_lists) == 2:
            filtered_sentences = doubleFilter(df3.iloc[i]['Processed_Sentences'], kw_lists[0], kw_lists[1])
            sentences_qn.append(filtered_sentences)
        elif len(kw_lists) == 1:
            filtered_sentences = singleFilter(df3.iloc[i]['Processed_Sentences'], kw_lists[0])
            sentences_qn.append(filtered_sentences)
        elif len(kw_lists) == 3:
            filtered_sentences = singleFilter(doubleFilter(df3.iloc[i]['Processed_Sentences'], kw_lists[0], kw_lists[1]), kw_lists[2])
            sentences_qn.append(filtered_sentences)
        else:
            print("There must be 1, 2 or 3 keyword lists")

    # Process and save raw sentences
    merged_sentences = []
    for ls in sentences_qn:
        merged = ""
        for s in ls:
            merged += " " + s
        merged = re.sub("\s?\d+\W?\s?", "", merged)
        merged = re.sub("##PAGE_BREAK##", "", merged)
        merged_sentences.append(merged)
    df4 = df3.copy()
    df4['relevant_sentences'] = merged_sentences
    df4 = df4[['IssuerName', 'Ticker', 'CountryOfIncorporation', 'GICSSector', 'GICSSubIndustry', 'relevant_sentences', qn_name]]

    # Remove stopwrods and formatting
    merged_sentences = []
    lemmatizer = WordNetLemmatizer()
    stopwords_en = set(stopwords.words('english'))
    for ls in sentences_qn:
        merged = ""
        for s in ls:
            merged += " " + s
        merged = re.sub("\s?\d+\W?\s?", "", merged)
        merged = re.sub("##PAGE_BREAK##", "", merged)
        tokens = merged.split(" ")
        tokens = [x for x in tokens if x not in stopwords_en]
        merged = " ".join([lemmatizer.lemmatize(token) for token in tokens])
        merged_sentences.append(merged)
    
    return merged_sentences

def getCountVectDf(qn_name, df3, merged_sentences):
    count_vectorizer = CountVectorizer(stop_words='english')
    X_count = count_vectorizer.fit_transform(merged_sentences)
    bow = X_count.toarray()
    target = df3[qn_name]
    matrix = np.hstack((bow, np.array(target).reshape(-1,1)))
    count_df = pd.DataFrame(matrix, columns=np.append(count_vectorizer.get_feature_names(), 'target_col'))
    return count_df

def getCountVectVaderDf(qn_name, df3, merged_sentences):
    # count vectorizer + vader
    count_vectorizer = CountVectorizer(stop_words='english')
    X_count = count_vectorizer.fit_transform(merged_sentences)
    bow = X_count.toarray()

    neg, neu, pos, compound = [], [], [], []
    sid_obj = SentimentIntensityAnalyzer()
    for s in merged_sentences:
        scores = sid_obj.polarity_scores(s)
        neg.append(scores['neg'])
        neu.append(scores['neu'])
        pos.append(scores['pos'])
        compound.append(scores['compound'])

    target = df3[qn_name]
    matrix = np.hstack((bow, np.array(neg).reshape(-1,1)))
    matrix = np.hstack((matrix, np.array(neu).reshape(-1,1)))
    matrix = np.hstack((matrix, np.array(pos).reshape(-1,1)))
    matrix = np.hstack((matrix, np.array(compound).reshape(-1,1)))
    matrix = np.hstack((matrix, np.array(target).reshape(-1,1)))
    column_names = np.append(count_vectorizer.get_feature_names(), ['neg_score', 'neu_score', 'pos_score', 'compound_score', 'target_col'])
    countVect_vader_df = pd.DataFrame(matrix, columns=column_names)
    return countVect_vader_df

def getTfidfDf(qn_name, df3, merged_sentences):
    # tfidf vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = tfidf_vectorizer.fit_transform(merged_sentences)
    tfidf_matrix = X_tfidf.toarray()
    target = df3[qn_name]
    tfidf_matrix = np.hstack((tfidf_matrix, np.array(target).reshape(-1,1)))
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=np.append(tfidf_vectorizer.get_feature_names(), 'target_col'))
    return tfidf_df

def getTfidfVaderDf(qn_name, df3, merged_sentences):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = tfidf_vectorizer.fit_transform(merged_sentences)
    bow = X_tfidf.toarray()

    neg, neu, pos, compound = [], [], [], []
    sid_obj = SentimentIntensityAnalyzer()
    for s in merged_sentences:
        scores = sid_obj.polarity_scores(s)
        neg.append(scores['neg'])
        neu.append(scores['neu'])
        pos.append(scores['pos'])
        compound.append(scores['compound'])

    target = df3[qn_name]
    matrix = np.hstack((bow, np.array(neg).reshape(-1,1)))
    matrix = np.hstack((matrix, np.array(neu).reshape(-1,1)))
    matrix = np.hstack((matrix, np.array(pos).reshape(-1,1)))
    matrix = np.hstack((matrix, np.array(compound).reshape(-1,1)))
    matrix = np.hstack((matrix, np.array(target).reshape(-1,1)))
    column_names = np.append(tfidf_vectorizer.get_feature_names(), ['neg_score', 'neu_score', 'pos_score', 'compound_score', 'target_col'])
    tfidf_vader_df = pd.DataFrame(matrix, columns=column_names)
    return tfidf_vader_df


def binaryModelPredict(qn_name, kw_lists, feature_engineering, model, df3):
    merged_sentences = getMergedSentences(qn_name, kw_lists, df3)
    df = feature_engineering(qn_name, df3, merged_sentences)
    df_train = df[df['target_col'].isna()==False]
    df_test = df[df['target_col'].isna()==True]
    x_train = df_train.drop(columns=['target_col'])
    y_train = df_train['target_col']
    model.fit(x_train, y_train)
    return model.predict(df_test.drop(columns=['target_col']))


def decisionTreeTrain(df_train, df_test):
    x_train = df_train.drop(columns=['target_col'])
    y_train = df_train['target_col']

    decision_tree=DecisionTreeClassifier(random_state=123)
    decision_tree.fit(x_train,y_train)
    acc=accuracy_score(y_train,decision_tree.predict(x_train))
    roc=roc_auc_score(y_train,decision_tree.predict(x_train))

    print("Training Accuracy: {:.3f} & ROC AUC Score: {:.3f}".format(acc, roc))

    return decision_tree.predict(df_test.drop(columns=['target_col']))


def randomForestTrain(df_train, df_test):
    x_train = df_train.drop(columns=['target_col'])
    y_train = df_train['target_col']

    random_forest=ensemble.RandomForestClassifier(random_state=123)
    random_forest.fit(x_train,y_train)
    acc=accuracy_score(y_train,random_forest.predict(x_train))
    roc=roc_auc_score(y_train,random_forest.predict(x_train))

    print("Training Accuracy: {:.3f} & ROC AUC Score: {:.3f}".format(acc, roc))

    return random_forest.predict(df_test.drop(columns=['target_col']))


def extraTreesTrain(df_train, df_test):
    x_train = df_train.drop(columns=['target_col'])
    y_train = df_train['target_col']

    et_classifier=ensemble.ExtraTreesClassifier(n_estimators=100, criterion='gini', random_state=123)
    et_classifier.fit(x_train,y_train)
    acc=accuracy_score(y_train,et_classifier.predict(x_train))
    roc=roc_auc_score(y_train,et_classifier.predict(x_train))

    print("Accuracy: {:.3f} & ROC AUC Score: {:.3f}".format(acc, roc))

    return et_classifier.predict(df_test.drop(columns=['target_col']))

def gradientBoostingTrain(df_train, df_test):
    # gradient boosting
    x_train = df_train.drop(columns=['target_col'])
    y_train = df_train['target_col']

    gradient_boosting=ensemble.GradientBoostingClassifier(random_state=123)
    gradient_boosting.fit(x_train,y_train)
    acc=accuracy_score(y_train,gradient_boosting.predict(x_train))
    roc=roc_auc_score(y_train,gradient_boosting.predict(x_train))

    print("Accuracy: {:.3f} & ROC AUC Score: {:.3f}".format(acc, roc))

    return gradient_boosting.predict(df_test.drop(columns=['target_col']))

def logisticRegressionTrain(df_train, df_test):
    x_train = df_train.drop(columns=['target_col'])
    y_train = df_train['target_col']

    logit=LogisticRegression(random_state=123)
    logit.fit(x_train,y_train)
    acc=accuracy_score(y_train,logit.predict(x_train))
    roc=roc_auc_score(y_train,logit.predict(x_train))

    print("Accuracy: {:.3f} & ROC AUC Score: {:.3f}".format(acc, roc))

    return logit.predict(df_test.drop(columns=['target_col']))


def randomForestMultiTrain(df_train, df_test):
    x_train = df_train.drop(columns=['target_col'])
    y_train = df_train['target_col']

    random_forest=ensemble.RandomForestClassifier(random_state=123)
    random_forest.fit(x_train,y_train)
    acc=accuracy_score(y_train,random_forest.predict(x_train))
    classes = random_forest.classes_
    y_proba = random_forest.predict_proba(x_train)
    roc_auc_ovr = {}
    for i in range(len(classes)):
        # Gets the class
        c = classes[i]
        
        # Prepares an auxiliar dataframe to help with the plots
        df_aux = x_train.copy()
        df_aux['class'] = [1 if y == c else 0 for y in y_train]
        df_aux['prob'] = y_proba[:, i]
        df_aux = df_aux.reset_index(drop = True)

        # Calculates the ROC AUC OvR
        roc_auc_ovr[c] = roc_auc_score(df_aux['class'], df_aux['prob'])

    avg_roc_auc = 0
    i = 0
    for k in roc_auc_ovr:
        avg_roc_auc += roc_auc_ovr[k]
        i += 1
        print(f"{k} ROC AUC OvR: {roc_auc_ovr[k]:.3f}")
        # print(f"average ROC AUC OvR: {avg_roc_auc/i:.3f}")  

    print("Accuracy: {:.3f} & ROC AUC Score: {:.3f}".format(acc, avg_roc_auc/i))

    return random_forest.predict(df_test.drop(columns=['target_col']))


def stratifiedKFoldModelEvaluation(models, feature_engineering, question_keywords, df, k=5):
    kf2 = StratifiedKFold(n_splits=k, random_state=None)
    model_results = {}
    for qn_name in question_keywords:
        model_results[qn_name] = []
        kw_lists = question_keywords[qn_name]
        merged_sentences = getMergedSentences(qn_name, kw_lists, df)
        for func in feature_engineering:
            temp_df = func(qn_name, df, merged_sentences)
            x = temp_df.drop(columns=['target_col'])
            y = temp_df['target_col']
            for model in models:
                result = cross_val_score(model, x, y, scoring="roc_auc_ovr", cv=kf2)
                if (np.isnan(result.mean())):
                    continue
                model_results[qn_name].append((func.__name__, model, result.mean()))
        model_results[qn_name].sort(key=lambda x: x[2], reverse=True)
        best = model_results[qn_name][0]
        print(f"Best weighted ROC AUC for {qn_name}: {best[2]}. Feature engineering: {best[0]}, Model: {best[1]} ")
    return model_results
