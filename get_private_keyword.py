import pandas as pd

file_name= "Data/Data.xlsx"
data = pd.read_excel(file_name, sheet_name="Question", skiprows=2)
pattern_list = data["Pattern Template"].tolist()

intent = 'hỏi_đáp_nghề_nghiệp'
list_keywords = []
for pattern in set(pattern_list):
    
    label = pattern.split("|")[0].strip()
    if label != intent:
        continue
    
    for keyword in  pattern.split("|")[1:]:
        if keyword not in list_keywords:
            list_keywords.append(keyword)

# for item in list_keywords:
    # print(item)