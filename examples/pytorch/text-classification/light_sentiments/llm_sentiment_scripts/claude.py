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

def get_result_by_palm2(prompt, candidate_count=1):
    req_data = { 
        "prompt":prompt,
        "channel":"insight",
        "candidate_count": candidate_count,
    }   
    url = "http://192.9.146.53:8039/api/v0/llm/palm/"
    res = requests.post(url=url,json=req_data)
    result_json = res.json()
    # print(result_json)
    if 'result' in result_json and result_json['result'] != "" and len(result_json["result"]) >= 1:
        return result_json['result']
    else:
        print(result_json)
        pass
    return None

def extract_contents(text):
    pattern = r'\[(.*?)\]'  # 匹配 [ 和 ] 之间的内容，括号内的 .*? 表示非贪婪模式匹配任意字符
    result = re.findall(pattern, text)  # 使用 re.findall 函数匹配所有符合条件的子串
    return result                                                                                                                                                                                   
results = []

# 首个订单的提示词
#2. The addition must consist of essential content and be limited to a maximum of 20 words.
TEXT = """
##instruction## 
This is a task about finding the user's emotion on each of the sentence user said in a conversation. Please check each sentence in the text in sequence and rate it's emotion if it is labeled as "[user]". User sentence means the sentence is labeled as "[user]", and advisor sentence mean the sentence is labeled as "[advisor]". Please rate the emotion of each user sentence on a scale of -1 to 1, where -1 represents a very negative user emotion, 0 represents a neutral user emotion, and 1 represents a very positive user emotion. User sentences with negative emotion should be rated smaller than 0, user sentences without obvious emotional expression should be scored close to 0, and user sentences with positive emotion should be rated bigger than 0. When rating each user's sentence, consider its overall tone, clarity, and effectiveness in conveying user thoughts, concerns, and emotions based on contextual semantics. Note that any sentence labeled as "[advisor]" can't be rated, but all sentences labeled as "[user]" must be rated.
####

##example input##
0. [advisor]: Hi, welcome, this is Eva💫
1. [user]: Hello
2. [user]: My name is Jada
3. [advisor]: Hi Jada nice to meet you
4. [advisor]: How are you doing today dear?
5. [user]: I could be better how is your day?
6. [advisor]: I am good thank you so much how can I help you today?
7. [user]: Fine I had a friend of 20 years pass away today so a little bit sad on top of a break up
8. [advisor]:  Oh I'm sorry to hear that
9. [advisor]: how can I help you sweetheart?
10. [user]: Just wondering who am I going to be with in a relationship because I am 37 about to be 38 so time is moving pretty quickly for me ya know and thanks that’s sweet
11. [advisor]: Is there anyone special that needs my attention?
12. [user]: Maybe someone from my passed I think about a lot his name is sawyer but that’s about it unless u know something I don’t lol
13. [advisor]: Can you provide the date of birth and names of both of you?
14. [advisor]: Let me connect better with energy
15. [advisor]: Are you there dear?
16. [user]: My name is Anna thomas 7-3-85 his name is Jason sawyer August 26, 1975
17. [advisor]: Okay let me check it out
18. [advisor]: I can see that there is still a strong connection between you
19. [advisor]: Show this person will come back
20. [user]: That’s it
21. [advisor]: Hope that can help you.
####

##example output##
{"1": 0, "2": 0, "5": 0.5, "7": -0.4, "10": -0.2, "12": -0.1, "16": 0, "20": 0.3}
####

the length of the output list equals to the number of the user appeared in the conversation, and the output list should be directly evaluated by Python eval. Note that any sentence labeled as "[advisor]" can't be rated, but all sentences labeled as "[user]" must be rated.
Now is you turn:

##input##
0. [advisor]: Hi, welcome, this is Sybil
1. [advisor]: How can I help you?
2. [user]: Hi Sybil
3. [user]: My name is Victoria
4. [advisor]: Hello 👋🏽
5. [advisor]: How are you doing today dear?
6. [user]: I am doing ok but I would be doing better if I knew if my current relationship was going to be a long term one
7. [advisor]: May I also have your name and date of birth please.
8. [user]: I love this man so much but there is an age gap
9. [advisor]: and this person's
10. [user]: Victoria NeSmith 04/15/1980
11. [user]: Ayoub Barca 10/12/1996
12. [advisor]: thank you for allowing me to connect with your souls
13. [user]: Ty
14. [advisor]: You and this person still seem to have some building to do for now, but I see a lot of potential for a long term relationship.
15. [advisor]: I do feel like it is a soulmate connection
16. [user]: U do
17. [advisor]: I feel this person thinks of you often & does want to grow in something committed
18. [user]: Awwww great
19. [user]: Does he love me
20. [advisor]: sometimes there will be a lot of ups and downs in this relationship, but I see permanent growth
21. [advisor]: yes he does. very deeply
####

##output##
"""
byte_lens = len(TEXT.encode())
print(f"lens : {len(TEXT)}, bytes lens: {byte_lens}")

test_count = 5
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    for i in range(1,1+test_count):
        # order_result = executor.submit(get_result_by_claude,TEXT)
        order_result = executor.submit(get_result_by_palm2,TEXT)
        results.append(order_result)

    for future in concurrent.futures.as_completed(results):
        res = future.result()
        
        # res = res.replace("[","\n[")
        # res = extract_contents(res)
        res = json.loads(f'{res[0]}')
        
        print(res)
        print("="*20)

print("end")
