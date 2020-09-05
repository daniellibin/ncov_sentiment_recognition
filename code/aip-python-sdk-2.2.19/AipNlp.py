from aip import AipNlp
import pandas as pd
import time
""" 你的 APPID AK SK """
APP_ID = '19147810'
API_KEY = 'ZUWGXfd1Yd5jMtrel1gsgzrZ'
SECRET_KEY = 'GVbroIyTnlKX26rLXDGBkrPWkC1eSytl'

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

ids = []

train = pd.read_csv("../data/nCoV_100k_train.labled.csv")
train = train[train["情感倾向"].isin(['-1','0','1'])]
train["情感倾向"] = train["情感倾向"].astype(int)
train["微博中文内容"] = train["微博中文内容"].astype(str)

i = 0
while i < len(train):
    time.sleep(0.2)
    try:
        text = train["微博中文内容"][i]
        """ 调用词法分析 """
        res = client.sentimentClassify(text)
        if res["items"][0]["sentiment"]-1 == train["情感倾向"][i]:
            ids.append(i)

        print(i,res["items"][0]["sentiment"]-1,train["情感倾向"][i])

    except Exception as e:
        print(i,e)

    i += 1

train = train.loc[ids,:]
train.to_csv("baidu_train.csv",index=False)
