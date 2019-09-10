"""
@author: Mihir Naresh Shah, Swapnil Sachin Shah
Description: Python code to extract subreddit and store in a json file
"""
import praw
import json
import pandas as pd

reddit = praw.Reddit(client_id='client_id', client_secret="client_secret",
                     password='password', user_agent='user_agent',
                     username='username')

subreddit_DS = reddit.subreddit('datascience')
subredditDS_top = subreddit_DS.top(limit=600)


subreddit_Fitness = reddit.subreddit('fitness')
subredditFitness_top = subreddit_Fitness.top(limit=600)


subreddit_GOT = reddit.subreddit('gameofthrones')
subredditGOT_top = subreddit_GOT.top(limit=600)

title_dict_DS = {"titles": [], "DataSet": "DataScience", "body":[]}
for x in subredditDS_top:
    if not x.stickied:
        title_dict_DS["titles"].append(x.title)
        title_dict_DS["body"].append(x.selftext)


title_dict_Fitness = {"titles": [], "DataSet": "Fitness", "body":[]}

for y in subredditFitness_top:
    if not y.stickied:
        title_dict_Fitness["titles"].append(y.title)
        title_dict_Fitness["body"].append(x.selftext)

title_dict_GOT = {"titles": [], "DataSet": "GOT", "body":[]}

for y in subredditGOT_top:
    if not y.stickied:
        title_dict_GOT["titles"].append(y.title)
        title_dict_GOT["body"].append(x.selftext)



DS = pd.DataFrame(title_dict_DS)
FN = pd.DataFrame(title_dict_Fitness)
MS = pd.DataFrame(title_dict_GOT)

DS.to_json("DataScience.json", orient='split')
FN.to_json("Fitness.json", orient='split')
MS.to_json("GOT.json", orient='split')
print(DS)
print(FN)
print(MS)
