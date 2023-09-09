import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('./reddit_data.csv')
    fields = ['username', 'subreddit']
    for field in fields:
        labels = pd.factorize(df[field])[0]
        kwargs = {field: labels}
        df = df.assign(**kwargs)
    df = df.sort_values(by='utc')
    df = df.iloc[:-200000, :]
    df.to_csv('reddit.tsv', sep='\t', encoding='utf-8', index=False)
