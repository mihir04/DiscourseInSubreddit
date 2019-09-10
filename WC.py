"""
@author: Mihir Naresh Shah, Swapnil Sachin Shah
Description: Word clouds made for each dataset
"""


import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def wc(x):
    """

    :param x:
    :return:
    plotting of word cloud
    """
    plt.imshow(x, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

def main():
    dataset1 = pd.read_json('DataScience.json', orient='split')
    dataset2 = pd.read_json('Fitness.json', orient='split')
    dataset3 = pd.read_json('GOT.json', orient='split')
    df1 = dataset1.titles.values
    df2 = dataset2.titles.values
    df3 = dataset3.titles.values
    wc(WordCloud().generate(str(df1)))
    wc(WordCloud().generate(str(df2)))
    wc(WordCloud().generate(str(df3)))

main()
