import requests
import re
import concurrent.futures
import json

def get_result_by_claude(prompt):
    data = {
        "prompt":prompt,
        "channel":"prompt test"
    }
    url = "http://192.168.12.226:5555/api/v0/summary/slack/"
    try_count = 10
    while try_count>0:
        res = requests.post(url=url,json=data)
        if res.json()['text'] != "":
            return res.json()['text']
        try_count -= 1
    return ""

def extract_contents(text):
    pattern = r'\[([-0-9.]+)\]'  # 匹配方括号中的数值
    matches = re.findall(pattern, text)  # 查找所有匹配项并返回数字的列表
    result = list(map(float, matches))  # 将字符串列表转换为浮点数列表
    return result                                                                                                                                                                                   
results = []

# 首个订单的提示词
#2. The addition must consist of essential content and be limited to a maximum of 20 words.
TEXT = """
##instruction## 
This is a task to discover the magnitude of user emotional change by comparing current conversations with historical conversations. Please check each sentence in the text in sequence, and focus on the emotion of the sentence if it is labeled as "[user]". User sentence means the sentence is labeled as "[user]", and advisor sentence mean the sentence is labeled as "[advisor]". Please measure the user emotion in historical conversations and current conversations separately, and then compare the magnitude of change in the current user emotion compared to historical user emotion. Please rate the magnitude of user emotional change on a scale of -1 to 1, where -1 represents a very negative emotional change, 0 represents no change in user emotion, and 1 represents a very positive emotional change. Negative change of user emotion should be rated smaller than 0, and positive change of user emotion should be rated bigger than 0. The greater the positive change in user emotion, the higher the rating, and the greater the negative change in user emotion, the lower the rating. When measuring the user emotions, please consider its overall tone, clarity, and effectiveness in conveying user thoughts, concerns, and emotions based on contextual semantics. 
####

##example input 1##
Historical conversations:
0. [advisor]: Hi, welcome, this is Eva💫
1. [user]: Hello
2. [user]: My name is Jada
3. [advisor]: Hi Jada nice to meet you
4. [advisor]: How are you doing today dear?
5. [user]: I could be better how is your day?
6. [advisor]: I am good thank you so much how can I help you today?

Current conversations:
7. [user]: Fine I had a friend of 20 years pass away today so a little bit sad on top of a break up
####

##example output 1##
[-0.8]
####

##example input 2##
Historical conversations:
0. [advisor]: Hi, sweet, how can i help you?
1. [user]: Hello
2. [user]: It’s my first time ever using this
3. [user]: And actually doing this
4. [advisor]: Welcome 🤗
5. [advisor]: You can ask the questions you want
6. [user]: Me and my long term boyfriend of 3 half years have broken hp
7. [user]: He broke up with me around 3 weeks ago now
8. [advisor]: I'm so sorry to hear this, darling

Current conversations:
9. [user]: Honestly our relationship was so good, every couple has their ups and downs but our arguments were not that bad it was just over small things
####

##example output 2##
[0.2]
####

Please focus mainly on the user's sentences and avoid being greatly influenced by the advisor's sentences. Note that the output must be a non empty list with a length of 1, which could be directly read by Python. Please only output the score list, do not make extensive textual statements.
Now is your turn:

##input##
Historical conversations:
0. [advisor]: can i have you name and day of birth
1. [user]: Hello
2. [advisor]: How can I help you
3. [user]: My name is Christiana
June 1998
4. [user]: Will marvel and I reunite?
5. [user]: Does he still love me?
6. [advisor]: nice to meet you Christiana
7. [advisor]: yes i do see if there will be a reconnection but not now, I feel the best time for this will be at the end of September beginning
8. [advisor]:  Yes he does love you
9. [advisor]:  But He’s going through a lot of emotional stress
10. [user]: What kind of emotional stress?
11. [user]: And where is he right now?
12. [advisor]:  He needs to work on himself
13. [advisor]:  He’s trying to focus on his career
14. [user]: How about me? Will I become what he needs?
15. [advisor]:  Yes I do see you are meant to be someone that helps him succeed
16. [advisor]:  Helping other people as well
17. [user]: Yes I do and I love to see people succeed but I’m not getting that in return
18. [advisor]:  People always try to take advantage of your kindness
19. [user]: I want marvel to be safe
And I will love if we come back together
20. [advisor]:  Because it comes with all your good things like money happiness success peace
21. [user]: Yes they always do take advantage of that
22. [advisor]:  I definitely feel like he will be more at peace when you’re both together
23. [advisor]:  That’s not fair
24. [user]: Definitely
25. [user]: I love him so much
26. [advisor]:  He deserves better than them taking advantage of him

Current conversations:
27. [user]: But he broke up with me
28. [user]: I wish he will come back some day
####

##output##
"""
byte_lens = len(TEXT.encode())
print(f"lens : {len(TEXT)}, bytes lens: {byte_lens}")

test_count = 5
with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
    for i in range(1,1+test_count):
        order_result = executor.submit(get_result_by_claude,TEXT)
        results.append(order_result)

    for future in concurrent.futures.as_completed(results):
        res = future.result()
        # res = res.replace("[","\n[")
        
        res_new = extract_contents(res)
        # res = json.loads(f'{{{res[0]}}}')
        
        print(res_new)
        print(res)
        print("="*20)

print("end")
