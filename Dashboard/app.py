import dataiku
import numpy as np
import pandas as pd
from flask import request


# Call the dataset from the Flow
question = dataiku.Dataset("Consolidated_Output")  # 63 rows

# Load dataframes
df_question = question.get_dataframe()
df_question = df_question.reset_index(drop=True)
df_question = df_question.replace(np.nan, "")

# Calculate transition plan score
qs = ["Q" + str(i) for i in range(8, 15)]


def calc_score(row):
    """
    calculates score
    """
    total = 0
    for q in qs:
        if row[q] == "Yes" or row[q] == "Established Carbon Transition Plan":
            total += 1
        elif row[q] == "Plans to Transition to Low Carbon Environment":
            total += 0.5
    return total


df_question['score'] = df_question.apply(calc_score, axis=1)

# Calculate percentile
df_question['percentile'] = round(df_question['score'].rank(pct=True) * 100)

# Get company details
df_company = df_question[['Company', 'URL', 'ISIN', 'Ticker',
                          'CountryOfIncorporation', 'GICSSector', 'GICSSubIndustry', 'Year']]

# Print statement available in the "Log" tab
print(df_question)
print(df_company)

# Make company name as key
res = {}
df_dict_list = df_company.to_dict('records')
for item in df_dict_list:
    res[item["Company"]] = {key: item[key] for key in item if key != "Company"}
res = json.dumps(res)
print(res)

# Get all sectors
sectors = df_company["GICSSector"].unique()
sectors = pd.DataFrame(sectors, columns=['sector']).to_json(orient="records")
sectors = json.dumps(sectors)
print(sectors)

# Get all sub-sectors
sub_sectors = df_company["GICSSubIndustry"].unique()
sub_sectors = pd.DataFrame(sub_sectors, columns=[
                           'sub_sector']).to_json(orient="records")
sub_sectors = json.dumps(sub_sectors)
print(sub_sectors)

# Get all countries
countries = df_company["CountryOfIncorporation"].unique()
countries = pd.DataFrame(
    countries, columns=['countries']).to_json(orient="records")
countires = json.dumps(countries)
print(countries)

# Get all year
years = df_company["Year"].unique()
years = pd.DataFrame(years, columns=['years']).to_json(orient="records")
years = json.dumps(years)
print(years)

# End points


@app.route('/')
def home():
    """
    Home routing, welcome page
    """
    return "Welcome"


@app.route('/dropdown')
def get_dropdown():
    """
    End point to get and populate all dropdowns
    """
    return json.dumps({"status": 200, "companies": res, "sectors": sectors, "sub_sectors": sub_sectors, "countries": countries, "years": years})


@app.route('/company')
def get_company():
    """
    End point to get the company's detail
    """
    return json.dumps({"status": 200, "companies": res})

# @app.route('/question/<year>', methods = ['GET'])
# def get_question(year):
#     """
#     End point to get answers to the question
#     """
#     year = request.args.get('year')
#     filtered_df = df_question.loc[(df['Year'] == year)]
#     ques = {}
#     for item in filtered_df.to_dict('records'):
#         ques[item['Company']] = {key: item[key] for key in item if key != "Company"}
#     ques = json.dumps(ques)
#     print("ques", ques)
#     return json.dumps({"status":200, "questions":ques})


@app.route('/question')
def get_question():
    """
    End point to get answers to the question
    """
    ques = {}
    for item in df_question.to_dict('records'):
        ques[item['Company']] = {key: item[key]
                                 for key in item if key != "Company"}
    ques = json.dumps(ques)
    print("ques", ques)
    return json.dumps({"status": 200, "questions": ques})


@app.route('/peer', methods=['GET'])
def get_peer():
    """
    End point to get peer scoring
    """
    return json.dumps({"status": 200, "questions": json.dumps(df_question.to_dict('records'))})

# @app.route('/peer', methods = ['GET'])
# def get_peer():
#     """
#     End point to get peer scoring
#     """
#     peerQ = ["Q3", "Q4", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14"]
#     year = request.args.get('year')
#     sector = request.args.get('sector')
#     country = request.args.get('country')
#     filtered_df = filtered_df.loc[df['Year'] == year]
#     if sector != "Sector":
#         filtered_df = filtered_df.loc[df['GICSSector'] == sector]
#     if sub_sector != "Sub-Sector":
#         filtered_df = filtered_df.loc[df['GICSSubIndustry'] == sub_sector]
#     if country != "Country":
#         filtered_df = filtered_df.loc[df['CountryOfIncorporation'] == country]
#     peer = {}
#     for q in peerQ:
#         if q != "Q8":
#             peer[q + "Peer"] = round((filtered_df[q] == "Yes").sum() / len(filtered_df[q]) * 100)
#         elif q == "Q8":
#             peer[q + "Peer"] = round((filtered_df[q] == "Established Carbon Transition Plan").sum() / len(filtered_df[q]) * 100)
#     peer = json.dumps(peer)
#     print("peer", peer)
#     return json.dumps({"status":200, "peers":peer})
