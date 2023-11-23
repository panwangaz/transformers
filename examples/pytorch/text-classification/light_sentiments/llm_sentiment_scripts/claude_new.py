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
    pattern = r'\[(.*?)\]'  # 匹配 [ 和 ] 之间的内容，括号内的 .*? 表示非贪婪模式匹配任意字符
    result = re.findall(pattern, text)  # 使用 re.findall 函数匹配所有符合条件的子串
    return result                                                                                                                                                                                   
results = []

# 首个订单的提示词
#2. The addition must consist of essential content and be limited to a maximum of 20 words.
TEXT = """
##instruction## 
This is a task to discover the magnitude of user emotional change by comparing current conversations with historical conversations. Please check each sentence in the text in sequence, and focus on the emotion of the sentence if it is labeled as "[user]". User sentence means the sentence is labeled as "[user]", and advisor sentence mean the sentence is labeled as "[advisor]". Please measure the user emotion in historical conversations and current conversations separately, and then compare the magnitude of change in the current user emotion compared to historical user emotion. Please rate the magnitude of user emotional change on a scale of -1 to 1, where -1 represents a very negative emotional change, 0 represents no change in user emotion, and 1 represents a very positive emotional change. Negative change of user emotion should be rated smaller than 0, and positive change of user emotion should be rated bigger than 0. The greater the positive change in user emotion, the higher the rating, and the greater the negative change in user emotion, the lower the rating. When measuring the user emotions, please consider its overall tone, clarity, and effectiveness in conveying user thoughts, concerns, and emotions based on contextual semantics.
####

##example input##
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
8. [advisor]:  Oh I'm sorry to hear that
9. [advisor]: how can I help you sweetheart?
####

##example output##
-0.9
####

The output must be a non empty number, which should be directly evaluated by Python eval.
Now is you turn:

##input##
Historical conversations:
0. [advisor]: Hi, welcome, this is Tarot Insights
1. [advisor]: How can I help you?
2. [user]: Hey my name is Teaira Hill my birthday is 12/08/2000 can I have a “ What I Need To Hear” type of reading?
3. [advisor]: Sure
4. [advisor]: I do just need a specific question
5. [user]: What I need to do like guidance
6. [advisor]: I would need it direct questions as well
7. [advisor]: So if you would like you could choose from one of my premium services
8. [advisor]: Or you can purchase a tarot card reading
9. [user]: Can I have the free reading
10. [advisor]:  You will need to go to my premium services and then choose one of my Tarot cards or a full life reading I offer both
11. [advisor]: You don’t want the free one every three minutes though
12. [advisor]: It’s not free

Current conversations:
13. [user]: If it’s not free I don’t want it I want the free one
14. [advisor]: Yes it’s only 3.99 but there’s no charge
15. [advisor]: You can either add coins or text me
16. [advisor]: Once you do, purchase coins
17. [user]: Can I have my reading
18. [advisor]: Yes what are your questions
19. [user]: Add more
20. [advisor]: What type of reading do you want
21. [user]: Give me more time
22. [user]: Guidance
23. [advisor]: You need to add credits
24. [advisor]: This is only three minutes
25. [advisor]: You got to buy coins
26. [advisor]: You can’t add credits until I get done with you
27. [advisor]: Are you there
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
        
        # res = extract_contents(res)
        # res = json.loads(f'{{{res[0]}}}')
        
        print(res)
        print("="*20)

print("end")
