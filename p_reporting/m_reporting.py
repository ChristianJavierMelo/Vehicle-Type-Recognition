import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# reporting functions

'''
def visualize_barplot(df,title):
    fig, ax = plt.subplots(figsize=(15,8))
    chart = sns.barplot(data=df, x='Make', y='Combined MPG')
    plt.title(title + "\n", fontsize=16)
    return chart

def visualize_lineplot(df,title):
    fig, ax = plt.subplots(figsize=(15,8))
    chart = sns.lineplot(data=df, x='Make', y='Combined MPG')
    plt.title(title + "\n", fontsize=16)
    return chart
'''

def plotting_function(df,title,args):
    fig, ax = plt.subplots(figsize=(16,8))
    plt.title(title + "\n", fontsize=16)
    if args.bar == True:
        sns.barplot(data=df, x='Make', y='Combined MPG')
        return fig
    elif args.line == True:
        sns.lineplot(data=df, x='Make', y='Combined MPG')
        return fig

def save_viz(fig,title):
    fig.savefig('./data/results/' + title + '.png')