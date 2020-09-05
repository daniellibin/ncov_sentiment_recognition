#百度通用翻译API,不包含词典、tts语音合成等资源，如有相关需求请联系translate_api@baidu.com
# coding=utf-8

import http.client
import hashlib
import urllib
import random
import json
import pandas as pd
import time
import requests

appid =   # 填写你的appid
secretKey =   # 填写你的密钥

httpClient = None

myurl = 'http://api.fanyi.baidu.com/api/trans/vip/translate'

fromLang = 'auto'   #原文语种
toLang = 'en'   #译文语种

train = pd.read_csv("data/train.csv")

content = list(train["微博中文内容"])


for i in range(len(train)):
    time.sleep(1)
    q = content[i] * 10
    salt = random.randint(32768, 65536)
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign

    try:
        res = requests.post(myurl)
        print(res)
        result_all = res.text
        result = json.loads(result_all)

        print(result)
        content[i] = result["trans_result"][0]["dst"]

    except Exception as e:
        print(e)

    finally:
        if httpClient:
            httpClient.close()

train["category"] = content
train.to_csv("data/translate.csv", index=False)


