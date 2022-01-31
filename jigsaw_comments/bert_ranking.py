# Asthetics
import warnings
import sklearn.exceptions

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# General
import glob
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import re
import random
import gc

pd.set_option('display.max_columns', None)
np.seterr(divide='ignore', invalid='ignore')
gc.enable()

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
# NLP
from transformers import AutoTokenizer, AutoModel

from data_utils import validate_correlation

# Random Seed Initialize
RANDOM_SEED = 42
os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

def seed_everything(seed=RANDOM_SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything()

# Device Optimization
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print(f'Using device: {device}')

# CFG
params = {
    'device': device,
    'debug': False,
    'checkpoint': 'roberta-base',
    'output_logits': 768,
    'max_len': 256,
    'batch_size': 16,
    'dropout': 0.2,
    'num_workers': 2,
    'epochs': 5,
    'lr': 2e-5,
    'margin': 0.7,
    'scheduler_name': 'OneCycleLR',
    'max_lr': 5e-5,                 # OneCycleLR
    'pct_start': 0.1,               # OneCycleLR
    'anneal_strategy': 'cos',       # OneCycleLR
    'div_factor': 1e3,              # OneCycleLR
    'final_div_factor': 1e3,        # OneCycleLR
    'no_decay': True
}

# Text Cleaning

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


def load_train_data(path):
    data_dir = path
    train_file_path = os.path.join(data_dir, 'validation_data_5_folds.csv')
    print(f'Train file: {train_file_path}')

    train_df = pd.read_csv(train_file_path)

    tqdm.pandas()
    train_df['less_toxic'] = train_df['less_toxic'].progress_apply(text_cleaning)
    train_df['more_toxic'] = train_df['more_toxic'].progress_apply(text_cleaning)

    train_df.sample(10)
    train_df.groupby(['kfold']).size()

    # kfold
    # 0    6022
    # 1    6022
    # 2    6022
    # 3    6021
    # 4    6021

    return train_df

# Dataset
class BERTDatasetPredictor:
    def __init__(self, text, max_len=params['max_len'], checkpoint=params['checkpoint']):
        self.text = text
        self.max_len = max_len
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.num_examples = len(self.text)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        text = str(self.text[idx])

        tokenized_text = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        ids = tokenized_text['input_ids']
        mask = tokenized_text['attention_mask']
        token_type_ids = tokenized_text['token_type_ids']

        return {'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)}

class BERTDataset:
    def __init__(self, more_toxic, less_toxic, max_len=params['max_len'], checkpoint=params['checkpoint']):
        self.more_toxic = more_toxic
        self.less_toxic = less_toxic
        self.max_len = max_len
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.num_examples = len(self.more_toxic)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        more_toxic = str(self.more_toxic[idx])
        less_toxic = str(self.less_toxic[idx])

        tokenized_more_toxic = self.tokenizer(
            more_toxic,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        tokenized_less_toxic = self.tokenizer(
            less_toxic,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_attention_mask=True,
            return_token_type_ids=True,
        )

        ids_more_toxic = tokenized_more_toxic['input_ids']
        mask_more_toxic = tokenized_more_toxic['attention_mask']
        token_type_ids_more_toxic = tokenized_more_toxic['token_type_ids']

        ids_less_toxic = tokenized_less_toxic['input_ids']
        mask_less_toxic = tokenized_less_toxic['attention_mask']
        token_type_ids_less_toxic = tokenized_less_toxic['token_type_ids']

        return {'ids_more_toxic': torch.tensor(ids_more_toxic, dtype=torch.long),
                'mask_more_toxic': torch.tensor(mask_more_toxic, dtype=torch.long),
                'token_type_ids_more_toxic': torch.tensor(token_type_ids_more_toxic, dtype=torch.long),
                'ids_less_toxic': torch.tensor(ids_less_toxic, dtype=torch.long),
                'mask_less_toxic': torch.tensor(mask_less_toxic, dtype=torch.long),
                'token_type_ids_less_toxic': torch.tensor(token_type_ids_less_toxic, dtype=torch.long),
                'target': torch.tensor(1, dtype=torch.float)}

# Scheduler

def get_scheduler(optimizer, scheduler_params=params):
    if scheduler_params['scheduler_name'] == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_params['T_0'],
            eta_min=scheduler_params['min_lr'],
            last_epoch=-1
        )
    elif scheduler_params['scheduler_name'] == 'OneCycleLR':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=scheduler_params['max_lr'],
            steps_per_epoch=int(df_train.shape[0] / params['batch_size']) + 1,
            epochs=scheduler_params['epochs'],
            pct_start=scheduler_params['pct_start'],
            anneal_strategy=scheduler_params['anneal_strategy'],
            div_factor=scheduler_params['div_factor'],
            final_div_factor=scheduler_params['final_div_factor'],
        )
    return scheduler

# Metrics

class MetricMonitor:
    def __init__(self, float_precision=4):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

# NLP Model

class ToxicityModel(nn.Module):
    def __init__(self, checkpoint=params['checkpoint'], params=params):
        super(ToxicityModel, self).__init__()
        self.checkpoint = checkpoint
        self.bert = AutoModel.from_pretrained(checkpoint, return_dict=False)
        self.layer_norm = nn.LayerNorm(params['output_logits'])
        self.dropout = nn.Dropout(params['dropout'])
        self.dense = nn.Sequential(
            nn.Linear(params['output_logits'], 128),
            nn.SiLU(),
            nn.Dropout(params['dropout']),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, token_type_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        preds = self.dense(pooled_output)
        return preds


# 1. Train Function

def train_fn(train_loader, model, criterion, optimizer, epoch, params, scheduler=None):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    
    for i, batch in enumerate(stream, start=1):
        ids_more_toxic = batch['ids_more_toxic'].to(device)
        mask_more_toxic = batch['mask_more_toxic'].to(device)
        token_type_ids_more_toxic = batch['token_type_ids_more_toxic'].to(device)
        ids_less_toxic = batch['ids_less_toxic'].to(device)
        mask_less_toxic = batch['mask_less_toxic'].to(device)
        token_type_ids_less_toxic = batch['token_type_ids_less_toxic'].to(device)
        target = batch['target'].to(device)

        logits_more_toxic = model(ids_more_toxic, token_type_ids_more_toxic, mask_more_toxic)
        logits_less_toxic = model(ids_less_toxic, token_type_ids_less_toxic, mask_less_toxic)
        loss = criterion(logits_more_toxic, logits_less_toxic, target)
        metric_monitor.update('Loss', loss.item())
        loss.backward()
        optimizer.step()
            
        if scheduler is not None:
            scheduler.step()
        
        optimizer.zero_grad()
        stream.set_description(f"Epoch: {epoch:02}. Train. {metric_monitor}")

# 2. Validate Function

def validate_fn(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    all_loss = []
    with torch.no_grad():
        for i, batch in enumerate(stream, start=1):
            ids_more_toxic = batch['ids_more_toxic'].to(device)
            mask_more_toxic = batch['mask_more_toxic'].to(device)
            token_type_ids_more_toxic = batch['token_type_ids_more_toxic'].to(device)
            ids_less_toxic = batch['ids_less_toxic'].to(device)
            mask_less_toxic = batch['mask_less_toxic'].to(device)
            token_type_ids_less_toxic = batch['token_type_ids_less_toxic'].to(device)
            target = batch['target'].to(device)

            logits_more_toxic = model(ids_more_toxic, token_type_ids_more_toxic, mask_more_toxic)
            logits_less_toxic = model(ids_less_toxic, token_type_ids_less_toxic, mask_less_toxic)
            loss = criterion(logits_more_toxic, logits_less_toxic, target)
            all_loss.append(loss.item())
            metric_monitor.update('Loss', loss.item())
            stream.set_description(f"Epoch: {epoch:02}. Valid. {metric_monitor}")
            
    return np.mean(all_loss)


def run_inference(test_df, models_dir, column_name='text'):
    predictions_nn, preds_list = None, []
    for model_name in glob.glob(models_dir + '/*.pth'):
        model = ToxicityModel()
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_name))
        model = model.to(params['device'])
        model.eval()

        test_dataset = BERTDatasetPredictor(
            text = test_df[column_name].values
        )
        test_loader = DataLoader(
            test_dataset, batch_size=params['batch_size'],
            shuffle=False, num_workers=params['num_workers'],
            pin_memory=True
        )

        temp_preds = None
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Predicting. '):
                ids= batch['ids'].to(device)
                mask = batch['mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                predictions = model(ids, token_type_ids, mask).to('cpu').numpy()
                
                if temp_preds is None:
                    temp_preds = predictions
                else:
                    temp_preds = np.vstack((temp_preds, predictions))

        if predictions_nn is None:
            predictions_nn = temp_preds
        else:
            predictions_nn += temp_preds
        
        preds_list.append(temp_preds)
        
    predictions_nn /= (len(glob.glob(models_dir + '/*.pth')))

    return predictions_nn, preds_list



if __name__ == '__main__':

    REMOTE = True
    TRAIN, VALIDATE = False, True
    DEVICE = torch.device('cuda') if REMOTE else 'cpu'

    if REMOTE:
        path = '/home/daca/kaggle_challenges/jigsaw_comments/data/'
    else:
        path = 'data/'

    
    # CFG
    params = {
        'device': device,
        'debug': False,
        'checkpoint': 'roberta-base',
        'output_logits': 768,
        'max_len': 256,
        'batch_size': 16,
        'dropout': 0.2,
        'num_workers': 2,
        'epochs': 5,
        'lr': 2e-5,
        'margin': 0.7,
        'scheduler_name': 'OneCycleLR',
        'max_lr': 5e-5,                 # OneCycleLR
        'pct_start': 0.1,               # OneCycleLR
        'anneal_strategy': 'cos',       # OneCycleLR
        'div_factor': 1e3,              # OneCycleLR
        'final_div_factor': 1e3,        # OneCycleLR
        'no_decay': True
    }
    
    if TRAIN:

        train_df = load_train_data(path)

        if params['debug']:
            train_df = train_df.sample(frac=0.01)
            print('Reduced training Data Size for Debugging purposes')

        best_models_of_each_fold = []

        gc.collect()
        for fold in range(train_df['kfold'].nunique()):
            print(f'******************** Training Fold: {fold+1} ********************')
            current_fold = fold
            df_train = train_df[train_df['kfold'] != current_fold].copy()
            df_valid = train_df[train_df['kfold'] == current_fold].copy()

            train_dataset = BERTDataset(
                df_train.more_toxic.values,
                df_train.less_toxic.values
            )
            valid_dataset = BERTDataset(
                df_valid.more_toxic.values,
                df_valid.less_toxic.values
            )

            train_dataloader = DataLoader(
                train_dataset, batch_size=params['batch_size'], shuffle=True,
                num_workers=params['num_workers'], pin_memory=True
            )
            valid_dataloader = DataLoader(
                valid_dataset, batch_size=params['batch_size']*2, shuffle=False,
                num_workers=params['num_workers'], pin_memory=True
            )
            
            model = ToxicityModel()
            model = torch.nn.DataParallel(model)
            model = model.to(params['device'])
            criterion = nn.MarginRankingLoss(margin=params['margin'])
            if params['no_decay']:
                param_optimizer = list(model.named_parameters())
                no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
                optimizer = optim.AdamW(optimizer_grouped_parameters, lr=params['lr'])
            else:
                optimizer = optim.AdamW(model.parameters(), lr=params['lr'])
            scheduler = get_scheduler(optimizer)

            # Training and Validation Loop
            best_loss = np.inf
            best_epoch = 0
            best_model_name = None
            for epoch in range(1, params['epochs'] + 1):
                train_fn(train_dataloader, model, criterion, optimizer, epoch, params, scheduler)
                valid_loss = validate_fn(valid_dataloader, model, criterion, epoch, params)
                if valid_loss <= best_loss:
                    best_loss = valid_loss
                    best_epoch = epoch
                    if best_model_name is not None:
                        os.remove(best_model_name)
                    torch.save(model.state_dict(), f"./output_bert/{params['checkpoint']}_{epoch}_epoch_f{fold+1}.pth")
                    best_model_name = f"./output_bert/{params['checkpoint']}_{epoch}_epoch_f{fold+1}.pth"

            # Print summary of this fold
            print('')
            print(f'The best LOSS: {best_loss} for fold {fold+1} was achieved on epoch: {best_epoch}.')
            print(f'The Best saved model is: {best_model_name}')
            best_models_of_each_fold.append(best_model_name)
            del df_train, df_valid, train_dataset, valid_dataset, train_dataloader, valid_dataloader, model
            _ = gc.collect()
            torch.cuda.empty_cache()

        for i, name in enumerate(best_models_of_each_fold):
            print(f'Best model of fold {i+1}: {name}')

    elif VALIDATE:

        # Test the quality of the model. Overfitted but just to see
        # val_data = pd.read_csv(os.path.join(path, "validation_data.csv"))
        # preds1, _ = predictions_nn, preds_list = run_inference(val_data, models_dir='./output_bert', column_name='less_toxic')
        # preds2, _ = predictions_nn, preds_list = run_inference(val_data, models_dir='./output_bert', column_name='more_toxic')
        # print(f'Validation Accuracy is { np.round((preds1 < preds2).mean() * 100,2)}')

        # val_data['p1'] = preds1.reshape(-1)
        # val_data['p2'] = preds2.reshape(-1)
        # val_data['diff'] = np.abs(preds2 - preds1)
        # val_data['correct'] = (preds1 < preds2).astype('int')
        ### Incorrect predictions with similar scores
        # val_data[val_data.correct == 0].sort_values('diff', ascending=True).head(30)

        # val_data.to_csv(os.path.join(path, "validation_pred_bert_rank.csv"), index=False)


        # Final Predictions
        test_df = pd.read_csv(os.path.join(path, "comments_to_score.csv"))
        tqdm.pandas()
        test_df['text'] = test_df['text'].progress_apply(text_cleaning)
        predictions_nn, preds_list = run_inference(test_df, models_dir='./output_bert')

        validate_correlation(pd.DataFrame(np.column_stack(preds_list)))

        # Build final prediction for kaggle
        sub_df = pd.DataFrame()
        sub_df['comment_id'] = test_df['comment_id']
        sub_df['score'] = predictions_nn
        sub_df['score'] = sub_df['score'].rank(method='first')
        sub_df.to_csv('submission.csv', index=False)

        print(sub_df.head(10))
        # pd.set_option('display.max_colwidth', None)
        pd.merge(sub_df, test_df, on=['comment_id']).sort_values('score', ascending=False)

    else:
        raise ValueError("Not a valid action")
