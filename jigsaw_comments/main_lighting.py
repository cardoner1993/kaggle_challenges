# https://www.kaggle.com/atamazian/nlp-getting-started-electra-pytorch-lightning/notebook
import os

from tqdm import tqdm

import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import torch

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# import our library
import torchmetrics

from transformers import BertModel, AdamW, BertTokenizerFast, AutoTokenizer, BertTokenizer

from sklearn.model_selection import KFold

from main import prepare_datasets

BATCH_SIZE = 64
EPOCHS = 10
MAX_LEN = 128
REMOTE = True


class LightningJigsawModel(LightningModule):
    def __init__(self, model_name, num_classes):
        super(LightningJigsawModel, self).__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(768, num_classes)
        # Optional add Relu final layer not used now in forward
        self.relu = nn.ReLU()
        # Loss function
        self.loss_function = nn.MSELoss()
        # Metrics
        self.metric = torchmetrics.MeanSquaredError()

    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask, return_dict=False)
        out = self.drop(out[1])
        outputs = self.fc(out)
        return outputs.type(torch.float64)

    def training_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
        z = self(b_input_ids, b_input_mask)
        loss = self.loss_function(z, b_labels)
        return loss

    def validation_step(self, batch, batch_idx):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]
        z = self(b_input_ids, b_input_mask)
        val_loss = self.loss_function(z, b_labels)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_mse_score', self.metric(z.reshape(-1), b_labels), prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.01, eps=1e-8)

    # Todo add test step if desired
    # def test_step(self, batch, batch_idx):
    #     x, y = batch.text[0].T, batch.label
    #     y_hat = self(x)
    #     loss = self.loss_function(y_hat, y)
    #     return dict(
    #         test_loss=loss,
    #         log=dict(
    #             test_loss=loss
    #         )
    #     )

    # def test_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     tensorboard_logs = dict(
    #         test_loss=avg_loss
    #     )
    #     return dict(
    #         avg_test_loss=avg_loss,
    #         log=tensorboard_logs
    #     )


class LitDataNLP(LightningDataModule):
    def __init__(self, fold, data_path, batch_size, tokenizer, max_length):
        super().__init__()
        self.fold = fold
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

    def setup(self, stage=None):
        # assumes data in format text and labels
        ruddit_data, toxic_data, toxic_multiling_data = \
            pd.read_csv(os.path.join(self.data_path, 'ruddit_with_text.csv')), \
            pd.read_csv(os.path.join(self.data_path, 'train.csv')), \
            pd.read_csv(os.path.join(self.data_path, 'jigsaw-toxic-comment-train.csv'))

        # Train process
        toxic_data, ruddit_data, toxic_multiling_data = prepare_datasets(ruddit_data, toxic_data, toxic_multiling_data)
        # Combine 3 datasets
        train_data = pd.concat([toxic_data, ruddit_data, toxic_multiling_data], ignore_index=True)

        text, labels = train_data.text.values.tolist(), train_data.y.values

        inputs = self.tokenizer.batch_encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )

        kf = KFold(5, shuffle=True, random_state=42)

        ids = np.array(inputs['input_ids'])
        mask = np.array(inputs['attention_mask'])

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

        self.train_inputs = torch.tensor(train_inputs)
        self.validation_inputs = torch.tensor(validation_inputs)
        self.train_labels = torch.tensor(train_labels, dtype=torch.float64)
        self.validation_labels = torch.tensor(validation_labels, dtype=torch.float64)
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
    for fold in range(5):
        # Load the best model per fold.
        model.load_state_dict(torch.load(f'{model_path}/fold_{fold}/model_best.ckpt')['state_dict'])
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

            preds = outputs[0]
            preds = preds.detach().cpu().numpy()

            # Store predictions and true labels
            predictions.append(preds)

        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        preds.append(flat_predictions)
    return np.round(np.mean(preds, axis=0), 0)


if __name__ == '__main__':

    TRAIN, VALIDATE = False, True

    if REMOTE:
        path = '/home/daca/kaggle_challenges/jigsaw_comments/data/'
    else:
        path = 'data/'

    DEVICE = 'cuda:7' if REMOTE else 'cpu'

    model_name = 'bert-base-uncased'
    model_path = './output_lightning'
    os.makedirs(model_path, exist_ok=True)

    if TRAIN:
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

        for fold in range(5):
            model_fold_path = os.path.join(model_path, f'fold_{fold}')
            os.makedirs(model_fold_path, exist_ok=True)
            dm = LitDataNLP(fold=fold, data_path=path, tokenizer=tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LEN)
            chk_callback = ModelCheckpoint(
                monitor='val_mse_score',
                dirpath=model_fold_path,
                filename='model_best',
                save_top_k=1,
                mode='min',
            )
            es_callback = EarlyStopping(
               monitor='val_mse_score',
               min_delta=0.001,
               patience=5,
               verbose=False,
               mode='min'
            )
            model = LightningJigsawModel(model_name=model_name, num_classes=1)

            if REMOTE:
                trainer = Trainer(
                    devices=[4, 5],
                    accelerator="gpu",
                    max_epochs=EPOCHS,
                    callbacks=[chk_callback, es_callback]
                )

            else:
                trainer = Trainer(
                    accelerator="cpu",
                    max_epochs=EPOCHS,
                    callbacks=[chk_callback, es_callback]
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
