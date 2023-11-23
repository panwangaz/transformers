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
This is a task to discover the magnitude of user emotional transition by comparing current conversations with historical conversations. Please check each sentence in the text in sequence, and focus on the emotion of the sentence if it is marked as "[user]". User sentence means the sentence is marked as "[user]", and advisor sentence means the sentence is makred as "[advisor]". Please measure the user emotion in historical conversations and current conversations separately, and then compare the magnitude of transition in the current user emotion compared to historical user emotion. Please rate the magnitude of user emotional transition on a scale of "-1" to "1", where "-1" represents a very negative emotional transition, "0" represents no transition in user emotion, and "1" represents a very positive emotional transition. When measuring the user emotions, please consider its overall tone, clarity, and effectiveness in conveying user thoughts, concerns, and emotions based on contextual semantics.
####

##attention##
1. Historical conversations mean the sentences are marked as "[Historical conversations]", current conversations mean the sentences are marked as "[Current conversations]". They are separated by a blank line.
2. Ignore some unclear implicit emotional transitions, such as users only making simple statements and providing personal information, etc, which should be rated between "-0.3" and "0.3".
3. Please focus mainly on the user's sentences and avoid being greatly influenced by the advisor's sentences.
4. The larger the positive transition, the closer the score is to "1"; the larger the negative transition, the closer the score is to "-1". Make sure that any emotional transition towards more negative must be rated as a value smaller than "0", and any emotional transition towards more positive must be rated as a value bigger than "0".
5. The output score must be enclosed in the "[]". Make sure "-" sign is contained in the output score and "[]" when there is any negative emotional transition.
####

##example input##
[Historical conversations]:
0. [advisor]: Hi, welcome, this is Eva💫
1. [user]: Hello
2. [user]: My name is Jada
3. [advisor]: Hi Jada nice to meet you
4. [advisor]: How are you doing today dear?
5. [user]: I could be better how is your day?
6. [advisor]: I am good thank you so much how can I help you today?

[Current conversations]:
7. [user]: Fine I had a friend of 20 years pass away today so a little bit sad on top of a break up
####

##example output##
[-0.8]
####

Note that the output must be a non empty list with a length of 1, which could be directly read by Python. Please only output the score list, don't make extensive textual statements. Now is your turn:

##input##
[Historical conversations]:
2. [advisor]: What do you want to know? Any specific questions? 
3. [user]: Not to bad… Hello 
4. [user]: Aye’Jae…3-2-80 
5. [advisor]: Hi welcome
6. [user]: I need help 
7. [advisor]: any specific questions?
8. [user]: I love this man so much but we have not been seeing eye to eye lately  
9. [user]: How long will we be mad 
10. [user]: When will he come back to me  
11. [user]: When will we be a family  
12. [advisor]: his name and date of birth plz
13. [user]: Emmanuel….11-14-86 
14. [advisor]: hello?
15. [advisor]: thanks 
16. [user]: .hello 
17. [user]: Your welcome  
18. [advisor]: sorry it shows some network error 
19. [advisor]: let me check 
20. [advisor]: It shows you do share a strong 
connection together 
21. [user]: Wow really  
22. [advisor]: I see a lot of chemistry attraction and interest there
23. [user]: Why are we going this 
24. [advisor]: shows he loves you very much

[Current conversations]:
25. [user]: It hurts like crazy
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
