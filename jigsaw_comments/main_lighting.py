# https://www.kaggle.com/atamazian/nlp-getting-started-electra-pytorch-lightning/notebook
from datetime import datetime
import os

from tqdm import tqdm

import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import torch

from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# import our library
import torchmetrics

from transformers import AdamW, BertTokenizerFast, AutoTokenizer, BertTokenizer, \
    get_linear_schedule_with_warmup, BertForSequenceClassification, BertModel

from sklearn.model_selection import KFold, train_test_split

from prepare_data import prepare_datasets, clean_data, text_cleaning, prepare_multiclass_datasets

BATCH_SIZE = 64
EPOCHS = 50
MAX_LEN = 128
REMOTE = True
TRAIN, VALIDATE = True, False
FOLDS = 1
DEVICE = 'cuda:7' if REMOTE else 'cpu'
DEBUG = False
model_type = "regression" # classification
monitor_metric = 'val_f1_score' if model_type == 'classification' else 'val_mse_score'


class LightningJigsawModel(LightningModule):
    def __init__(self, model_name, num_classes=None, model_type='regression'):
        super(LightningJigsawModel, self).__init__()
        self.model_type = model_type
        
        self.model_regr = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes, output_attentions=False,
                                                                           output_hidden_states=False)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(768, num_classes)
        # Optional add Relu final layer not used now in forward
        self.relu = nn.ReLU()
        
        self.model_regr._init_weights(self.fc)

        self.model_classification = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes,
                                                                           output_attentions=False,
                                                                           output_hidden_states=False)

        
        # Warm up wait n epochs
        self.warmup_steps = 0
        # Loss function
        self.loss_function = nn.MSELoss() if self.model_type == 'regression' else None
        # Metrics
        self.regr_metric = torchmetrics.MeanSquaredError()
        # threshold (float) – Threshold for transforming probability or logit predictions to binary (0,1) 
        self.classif_metric = torchmetrics.F1(num_classes=num_classes)

    def forward(self, ids, mask, b_labels=None):
        if self.model_type == 'regression':
            outputs = self.model_regr(input_ids=ids, attention_mask=mask, labels=b_labels)
            # out = self.drop(out[1])
            # out = self.fc(out)
            # outputs = self.relu(out)
            # return outputs.type(torch.float64)
            return outputs
        else:
            outputs = self.model_classification(input_ids=ids, attention_mask=mask, labels=b_labels)
            # return outputs.type(torch.float64)
            return outputs

    def on_train_batch_start(self, batch, batch_idx):
        # Logs learning
        self.log('learning_rate', self.trainer.lr_schedulers[0]['scheduler'].get_lr()[0], prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
        z = self(b_input_ids, b_input_mask, b_labels)

        if self.model_type == 'regression':
            # loss = self.loss_function(z, b_labels)
            loss = z[0]
            self.log('train_mse_score', loss, prog_bar=True, logger=True)
            # self.log('train_mse_score', self.metric(z.reshape(-1), b_labels), prog_bar=True, logger=True)      
        else:
            loss, logits = z[0], z[1]
            self.log('train_f1_score', self.classif_metric(logits, b_labels), prog_bar=True)

        self.log('train_loss', loss, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
        z = self(b_input_ids, b_input_mask, b_labels)
        
        if self.model_type == 'regression':
            # val_loss = self.loss_function(z, b_labels)
            val_loss = z[0]
            self.log('val_mse_score', val_loss, prog_bar=True, logger=True)
            # self.log('val_mse_score', self.metric(z.reshape(-1), b_labels), prog_bar=True, logger=True)
        else:
            val_loss, logits = z[0], z[1]
            self.log('val_f1_score', self.classif_metric(logits, b_labels), prog_bar=True)

        self.log('val_loss', val_loss, prog_bar=True)

        return val_loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=0.01, eps=1e-8)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]


class LitDataNLP(LightningDataModule):
    def __init__(self, fold, data_path, batch_size, tokenizer, max_length, debugging, model_type='regression'):
        super().__init__()
        self.fold = fold
        self.debugging = debugging
        self.batch_size = batch_size
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.train_inputs = None
        self.validation_inputs = None
        self.train_labels = None
        self.validation_labels = None
        self.train_masks = None
        self.validation_masks = None
        self.num_labels = None
        self.model_type = model_type

        self.train_data = self.prepare_dataset()
    
    def prepare_dataset(self):
        ruddit_data, toxic_data, toxic_multiling_data = \
            pd.read_csv(os.path.join(self.data_path, 'ruddit_with_text.csv')), \
            pd.read_csv(os.path.join(self.data_path, 'train.csv')), \
            pd.read_csv(os.path.join(self.data_path, 'jigsaw-toxic-comment-train.csv'))

        if self.model_type == 'regression':
            self.num_labels = 1
            # Train process
            toxic_data, ruddit_data, toxic_multiling_data = prepare_datasets(ruddit_data, toxic_data, toxic_multiling_data)
            # Combine 3 datasets
            # train_data = pd.concat([toxic_data, ruddit_data, toxic_multiling_data], ignore_index=True)
            train_data = pd.concat([toxic_data, ruddit_data], ignore_index=True)
        else:
            toxic_data, toxic_multiling_data = prepare_multiclass_datasets(toxic_data, toxic_multiling_data)
            # train_data = pd.concat([toxic_data, toxic_multiling_data], ignore_index=True)
            train_data = toxic_data
            self.num_labels = len(train_data['y'].unique())
        
        
        if self.debugging:
            train_data = train_data.sample(n=1000, random_state=1)
        
        return train_data

    def prepare_train_valid(self, folds, inputs, labels):

        ids = np.array(inputs['input_ids'])
        mask = np.array(inputs['attention_mask'])

        if folds <= 1:
            train_inputs, validation_inputs, train_masks, validation_masks, train_labels, validation_labels = \
                train_test_split(ids, mask, labels, test_size=0.2)
        else:
            kf = KFold(FOLDS, shuffle=True, random_state=42)

            # Check it out. The code implements kf cross validation in this way ?¿
            for fold, (tr_idx, val_idx) in enumerate(kf.split(ids, labels)):
                train_inputs = ids[tr_idx]
                train_labels = labels[tr_idx]
                validation_inputs = ids[val_idx]
                validation_labels = labels[val_idx]
                if fold == self.fold:
                    break

            for fold, (tr_idx, val_idx) in enumerate(kf.split(mask, labels)):
                train_masks = mask[tr_idx]
                validation_masks = mask[val_idx]
                if fold == self.fold:
                    break

        return train_inputs, validation_inputs, train_masks, validation_masks, train_labels, validation_labels

    def setup(self, stage=None):
        # assumes data in format text and labels
        # ruddit_data, toxic_data, toxic_multiling_data = \
        #     pd.read_csv(os.path.join(self.data_path, 'ruddit_with_text.csv')), \
        #     pd.read_csv(os.path.join(self.data_path, 'train.csv')), \
        #     pd.read_csv(os.path.join(self.data_path, 'jigsaw-toxic-comment-train.csv'))

        # if self.model_type == 'regression':
        #     self.num_labels = 1
        #     # Train process
        #     toxic_data, ruddit_data, toxic_multiling_data = prepare_datasets(ruddit_data, toxic_data, toxic_multiling_data)
        #     # Combine 3 datasets
        #     train_data = pd.concat([toxic_data, ruddit_data, toxic_multiling_data], ignore_index=True)
        #     # train_data = pd.concat([toxic_data, toxic_multiling_data], ignore_index=True)
        # else:
        #     toxic_data, toxic_multiling_data = prepare_multiclass_datasets(toxic_data, toxic_multiling_data)
        #     train_data = pd.concat([toxic_data, toxic_multiling_data], ignore_index=True)
        #     self.num_labels = len(train_data['y'].unique())

        # if self.debugging:
        #     train_data = train_data.sample(n=1000, random_state=1)

        # Further data cleaning.
        # train_data = clean_data(train_data, 'text')

        self.train_data['text'] = self.train_data['text'].apply(text_cleaning)

        text, labels = self.train_data.text.values.tolist(), self.train_data.y.values

        inputs = self.tokenizer.batch_encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )

        train_inputs, validation_inputs, train_masks, validation_masks, train_labels, validation_labels = \
            self.prepare_train_valid(FOLDS, inputs, labels)

        if self.model_type == 'regression':
            self.train_inputs = torch.tensor(train_inputs)
            self.validation_inputs = torch.tensor(validation_inputs)
            self.train_labels = torch.tensor(train_labels, dtype=torch.float)
            self.validation_labels = torch.tensor(validation_labels, dtype=torch.float)
            self.train_masks = torch.tensor(train_masks, dtype=torch.long)
            self.validation_masks = torch.tensor(validation_masks, dtype=torch.long)
        else:
            self.train_inputs = torch.tensor(train_inputs)
            self.validation_inputs = torch.tensor(validation_inputs)
            self.train_labels = torch.tensor(train_labels, dtype=torch.long)
            self.validation_labels = torch.tensor(validation_labels, dtype=torch.long)
            self.train_masks = torch.tensor(train_masks, dtype=torch.long)
            self.validation_masks = torch.tensor(validation_masks, dtype=torch.long)

    def train_dataloader(self):
        train_data = TensorDataset(self.train_inputs, self.train_masks, self.train_labels)
        train_sampler = RandomSampler(train_data)
        return DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

    def val_dataloader(self):
        validation_data = TensorDataset(self.validation_inputs, self.validation_masks, self.validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        return DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size)


def run_inference(data_dir, colname, model, device, model_path, batch_size: int = 32):
    test_df = pd.read_csv(data_dir)
    comments = test_df[colname].values.tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    indices = tokenizer.batch_encode_plus(comments, max_length=128, add_special_tokens=True,
                                          return_attention_mask=True, pad_to_max_length=True,
                                          truncation=True)
    input_ids = indices["input_ids"]
    attention_masks = indices["attention_mask"]

    test_inputs = torch.tensor(input_ids)
    test_masks = torch.tensor(attention_masks)

    # Create the DataLoader.
    test_data = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    print('Predicting labels...')

    preds = []
    for fold in range(FOLDS):
        # Load the best model per fold.
        model.load_state_dict(torch.load(f'{model_path}/fold_{fold}/model_best.ckpt', map_location=device)['state_dict'])
        model.eval()
        model.to(device)

        # Tracking variables
        predictions = []

        # Predict
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch

            with torch.no_grad():
                outputs = model(b_input_ids, b_input_mask)

            preds = outputs.reshape(-1)
            preds = preds.detach().cpu().numpy()

            # Store predictions and true labels
            predictions.append(preds)

        flat_predictions = [item for sublist in predictions for item in sublist]
        preds.append(flat_predictions)
    return np.round(np.mean(preds, axis=0), 0)


if __name__ == '__main__':

    if REMOTE:
        path = '/home/daca/kaggle_challenges/jigsaw_comments/data/'
    else:
        path = 'data/'

    experiment_id = f"type_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    model_name = 'bert-base-uncased'
    model_path = os.path.join('./output_lightning', experiment_id)
    log_dir = os.path.join("logs", experiment_id)

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if TRAIN:
        tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

        for fold in range(FOLDS):
            model_fold_path = os.path.join(model_path, f'fold_{fold}')
            os.makedirs(model_fold_path, exist_ok=True)
            dm = LitDataNLP(fold=fold, data_path=path, tokenizer=tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LEN,
                            debugging=DEBUG, model_type=model_type)
            
            lr_monitor = LearningRateMonitor(logging_interval='step')
            
            chk_callback = ModelCheckpoint(
                monitor=monitor_metric,
                dirpath=model_fold_path,
                filename='model_best',
                save_top_k=1,
                mode='min',
            )
            
            es_callback = EarlyStopping(
               monitor=monitor_metric,
               min_delta=0.001,
               patience=5,
               verbose=False,
               mode='min'
            )
            model = LightningJigsawModel(model_name=model_name, num_classes=dm.num_labels, model_type=model_type)

            tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir)

            if REMOTE:
                trainer = Trainer(
                    # devices=[4, 5], error with scheduler from transformers if using more than one
                    devices=[4],
                    accelerator="gpu",
                    max_epochs=EPOCHS,
                    callbacks=[chk_callback, es_callback, lr_monitor],
                    logger=tb_logger
                )

            else:
                trainer = Trainer(
                    accelerator="cpu",
                    max_epochs=EPOCHS,
                    callbacks=[chk_callback, es_callback, lr_monitor],
                    logger=tb_logger
                )

            trainer.fit(model, dm)

    elif VALIDATE:
        model = LightningJigsawModel(model_name=model_name, num_classes=1)
        valid_path = "data/validation_data.csv"
        preds_less = run_inference(valid_path, 'less_toxic', model, device=DEVICE, model_path=model_path)
        preds_more = run_inference(valid_path, 'more_toxic', model, device=DEVICE, model_path=model_path)

        print(f'Validation Accuracy is {np.round((preds_less < preds_more).mean() * 100, 2)}')

    else:
        raise ValueError("Not a valid action")
