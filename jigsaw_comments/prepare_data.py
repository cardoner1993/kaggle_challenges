
def prepare_datasets(ruddit_data=None, toxic_data=None, toxic_multil_data=None):
    # Give more weight to severe toxic
    if toxic_data is not None:
        toxic_data['severe_toxic'] = toxic_data.severe_toxic * 2
        toxic_data['y'] = (toxic_data[['toxic', 'severe_toxic', 'obscene', 'threat',
                                       'insult', 'identity_hate']].sum(axis=1)).astype(int)
        toxic_data['y'] = toxic_data['y'] / toxic_data['y'].max()
        toxic_data = toxic_data[['comment_text', 'y']].rename(columns={'comment_text': 'text'})

    if ruddit_data is not None:
        # Ruddit data
        ruddit_data = ruddit_data[['txt',
                                   'offensiveness_score']].rename(columns={'txt': 'text', 'offensiveness_score': 'y'})
        ruddit_data['y'] = (ruddit_data['y'] - ruddit_data.y.min()) / (ruddit_data.y.max() - ruddit_data.y.min())

    if toxic_multil_data is not None:
        # Toxic multilingual
        toxic_multil_data['severe_toxic'] = toxic_multil_data.severe_toxic * 2
        toxic_multil_data['y'] = \
            (toxic_multil_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
                                'identity_hate']].sum(axis=1)).astype(int)
        toxic_multil_data['y'] = toxic_multil_data['y'] / toxic_multil_data['y'].max()
        toxic_multil_data = toxic_multil_data[['comment_text', 'y']].rename(columns={'comment_text': 'text'})

    # Binary problem
    # print("Describe toxic data:")
    # print(toxic_data['y'].describe())

    # print("Describe ruddit data:")
    # print(toxic_data['y'].describe())

    # print("Describe toxic multilabel data:")
    # print(toxic_data['y'].describe())

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
