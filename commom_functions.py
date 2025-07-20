import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

def pareto_analysis(df, col, filter_col=None, filter_val=None):
    """
    Creates a Pareto table showing counts, percentage, and cumulative percentage
    for a given column, optionally filtered by another column.
    """
    if filter_col and filter_val:
        df = df[df[filter_col] == filter_val]
    
    counts = df[col].value_counts()
    result = counts.to_frame(name='count')
    result['percentage'] = (result['count'] / result['count'].sum()) * 100
    result['cumulative_percentage'] = result['percentage'].cumsum()
    
    return result


def clean_textlines(df, columns):
    '''
    Cleans specified text columns in a dataframe by:
    - Replacing missing values with empty strings
    - Replacing newline characters ('\n') with spaces
    '''

    for col in columns:
        df[col] = df[col].fillna("").str.replace("\n", " ")
    
    return df


def remove_stopwords(text):
    '''
    Removes English stop words from the given text.
    '''

    if pd.isna(text):
        return " "
    
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    return " ".join(filtered_words)