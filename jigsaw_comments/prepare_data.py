from pydoc import describe
import re
from bs4 import BeautifulSoup


def prepare_multiclass_datasets(toxic_data=None, toxic_multil_data=None):
    # Give more weight to severe toxic
    if toxic_data is not None:
        toxic_data['severe_toxic'] = toxic_data.severe_toxic * 2
        toxic_data['y'] = (toxic_data[['toxic', 'severe_toxic', 'obscene', 'threat',
                                       'insult', 'identity_hate']].sum(axis=1)).astype(int)

        toxic_data = toxic_data[['comment_text', 'y']].rename(columns={'comment_text': 'text'})

    if toxic_multil_data is not None:
        # Toxic multilingual
        toxic_multil_data['severe_toxic'] = toxic_multil_data.severe_toxic * 2
        toxic_multil_data['y'] = \
            (toxic_multil_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
                                'identity_hate']].sum(axis=1)).astype(int)

        toxic_multil_data = toxic_multil_data[['comment_text', 'y']].rename(columns={'comment_text': 'text'})
    
    # Describe data by level
    print("Describe toxic data before transform:")
    print(toxic_data['y'].value_counts() / len(toxic_data))
    print("\nDescribe toxic multilabel data before transform:")
    print(toxic_multil_data['y'].value_counts() / len(toxic_multil_data))

    return toxic_data, toxic_multil_data


def prepare_datasets(ruddit_data=None, toxic_data=None, toxic_multil_data=None):
    
    # Give more weight to severe toxic
    if toxic_data is not None:
        toxic_data['severe_toxic'] = toxic_data.severe_toxic * 2
        toxic_data['y'] = (toxic_data[['toxic', 'severe_toxic', 'obscene', 'threat',
                                       'insult', 'identity_hate']].sum(axis=1)).astype(int)

        print("Describe toxic data before transform:")
        print(toxic_data['y'].describe())
        print(toxic_data['y'].unique())

        toxic_data['y'] = toxic_data['y'] / toxic_data['y'].max()
        toxic_data = toxic_data[['comment_text', 'y']].rename(columns={'comment_text': 'text'})

    if ruddit_data is not None:
        # Ruddit data
        ruddit_data = ruddit_data[['txt',
                                   'offensiveness_score']].rename(columns={'txt': 'text', 'offensiveness_score': 'y'})

        print("\nDescribe ruddit data before transform:")
        print(ruddit_data['y'].describe())
        print(ruddit_data['y'].unique())
        
        ruddit_data['y'] = (ruddit_data['y'] - ruddit_data.y.min()) / (ruddit_data.y.max() - ruddit_data.y.min())

    if toxic_multil_data is not None:
        # Toxic multilingual
        toxic_multil_data['severe_toxic'] = toxic_multil_data.severe_toxic * 2
        toxic_multil_data['y'] = \
            (toxic_multil_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
                                'identity_hate']].sum(axis=1)).astype(int)
        
        print("\nDescribe toxic multilabel data before transform:")
        print(toxic_multil_data['y'].describe())
        print(toxic_multil_data['y'].unique())

        toxic_multil_data['y'] = toxic_multil_data['y'] / toxic_multil_data['y'].max()
        toxic_multil_data = toxic_multil_data[['comment_text', 'y']].rename(columns={'comment_text': 'text'})

    # Binary problem
    print("\nDescribe toxic data:")
    print(toxic_data['y'].describe())

    print("Describe ruddit data:")
    print(ruddit_data['y'].describe())

    print("Describe toxic multilabel data:")
    print(toxic_multil_data['y'].describe())


    return toxic_data, ruddit_data, toxic_multil_data


def clean_data(data, col):
    # Clean some punctutations
    data[col] = data[col].str.replace('\n', ' \n ')
    data[col] = data[col].str.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)', r'\1 \2 \3')
    # Replace repeating characters more than 3 times to length of 3
    data[col] = data[col].str.replace(r'([*!?\'])\1\1{2,}', r'\1\1\1')
    # Add space around repeating characters
    data[col] = data[col].str.replace(r'([*!?\']+)', r' \1 ')
    # patterns with repeating characters
    data[col] = data[col].str.replace(r'([a-zA-Z])\1{2,}\b', r'\1\1')
    data[col] = data[col].str.replace(r'([a-zA-Z])\1\1{2,}\B', r'\1\1\1')
    data[col] = data[col].str.replace(r'[ ]{2,}', ' ').str.strip()

    return data


def text_cleaning(text):
    '''
    Cleans text into a basic form for NLP. Operations include the following:-
    1. Remove special charecters like &, #, etc
    2. Removes extra spaces
    3. Removes embedded URL links
    4. Removes HTML tags
    5. Removes emojis
    
    text - Text piece to be cleaned.
    '''
    template = re.compile(r'https?://\S+|www\.\S+') #Removes website links
    text = template.sub(r'', text)
    
    soup = BeautifulSoup(text, 'lxml') #Removes HTML tags
    only_text = soup.get_text()
    text = only_text
    
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    text = re.sub(r"[^a-zA-Z\d]", " ", text) #Remove special Charecters
    text = re.sub(' +', ' ', text) #Remove Extra Spaces
    text = text.strip() # remove spaces at the beginning and at the end of string

    return text
