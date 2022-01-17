import os
import gc
import random

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For Transformer Models
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast, BertModel, AdamW, \
    get_linear_schedule_with_warmup

# Utils
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
gpu_id = 5

CONFIG_TEST = dict(
    seed=42,
    model_name='bert-base-uncased',
    test_batch_size=64,
    max_length=128,
    num_classes=1,
    device=torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
)

folds_path = 'input/pytorch-w-b-jigsaw/'

# Train
CONFIG_TRAIN = dict(
    seed=42,
    model_name='bert-base-uncased',
    train_batch_size=64,
    max_length=128,
    num_classes=1,
    device=torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
)

CONFIG_TRAIN["tokenizer"] = BertTokenizerFast.from_pretrained(CONFIG_TRAIN['model_name'], do_lower_case=True)


def prepare_datasets(ruddit_data, toxic_data, toxic_multil_data):
    # Give more weight to severe toxic
    toxic_data['severe_toxic'] = toxic_data.severe_toxic * 2
    toxic_data['y'] = (toxic_data[['toxic', 'severe_toxic', 'obscene', 'threat',
                                   'insult', 'identity_hate']].sum(axis=1)).astype(int)
    toxic_data['y'] = toxic_data['y'] / toxic_data['y'].max()
    toxic_data = toxic_data[['comment_text', 'y']].rename(columns={'comment_text': 'text'})
    # Ruddit data
    ruddit_data = ruddit_data[['txt',
                               'offensiveness_score']].rename(columns={'txt': 'text', 'offensiveness_score': 'y'})
    ruddit_data['y'] = (ruddit_data['y'] - ruddit_data.y.min()) / (ruddit_data.y.max() - ruddit_data.y.min())
    # Toxic multilingual
    toxic_multil_data['severe_toxic'] = toxic_multil_data.severe_toxic * 2
    toxic_multil_data['y'] = \
        (toxic_multil_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
                            'identity_hate']].sum(axis=1)).astype(int)
    toxic_multil_data['y'] = toxic_multil_data['y'] / toxic_multil_data['y'].max()
    toxic_multil_data = toxic_multil_data[['comment_text', 'y']].rename(columns={'comment_text': 'text'})

    return toxic_data, ruddit_data, toxic_multil_data


def set_seed(seed=42):
    '''
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


class JigsawDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, labels=None):
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.text = text.values
        if labels is not None:
            self.labels = labels.values
        else:
            self.labels = None

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        response = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long)
        }

        if self.labels is not None:
            label = self.labels[index]
            response['label'] = torch.tensor(label, dtype=torch.float64)

        return response


class JigsawModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super(JigsawModel, self).__init__()
        # self.model = AutoModel.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(768, num_classes)
        # Optional add Relu final layer not used now in forward
        self.relu = nn.ReLU()

    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask, return_dict=False)
        out = self.drop(out[1])
        outputs = self.fc(out)
        return outputs.type(torch.float64)


@torch.no_grad()
def valid_fn(model, dataloader, device):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    PREDS = []

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)

        outputs = model(ids, mask)
        PREDS.append(outputs.view(-1).cpu().detach().numpy())

    PREDS = np.concatenate(PREDS)
    gc.collect()

    return PREDS


def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


# If can't debug the dataset then just update the num workers to 0 or main worker
def train_fn(dataset, device, learning_rate, epochs, k_folds=5):
    criterion = nn.MSELoss()
    criterion = criterion.cuda()
    input_data, y = dataset['text'], dataset['y']

    # train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_data, y,
    #                                                                                     random_state=2018,
    #                                                                                     test_size=0.2)
    #
    # train_dataset = JigsawDataset(train_inputs, CONFIG_TRAIN['tokenizer'], max_length=CONFIG_TRAIN['max_length'],
    #                               labels=train_labels)
    # train_loader = DataLoader(train_dataset, batch_size=CONFIG_TRAIN['train_batch_size'], shuffle=False)
    # val_dataset = JigsawDataset(validation_inputs, CONFIG_TRAIN['tokenizer'], max_length=CONFIG_TRAIN['max_length'],
    #                             labels=validation_labels)
    # val_loader = DataLoader(val_dataset, batch_size=CONFIG_TRAIN['train_batch_size'], shuffle=False)

    dataset = JigsawDataset(input_data, CONFIG_TRAIN['tokenizer'], max_length=CONFIG_TRAIN['max_length'],
                            labels=y)
    # dataset.tokenizer.save_pretrained(CONFIG_TEST['model_name'])

    # K-fold Cross Validation model evaluation
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=k_folds, shuffle=True)
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'Starting FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # Define data loaders for training and testing data in this fold
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG_TRAIN['train_batch_size'],
                                                   sampler=train_subsampler)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=CONFIG_TRAIN['train_batch_size'],
                                                 sampler=test_subsampler)

        total_steps = len(train_loader) * epochs
        # Implements Adam algorithm with weight decay fix as introduced in Decoupled Weight Decay Regularization.
        model = JigsawModel(CONFIG_TRAIN['model_name'], CONFIG_TRAIN['num_classes'])
        # model.apply(reset_weights)
        model = model.to(device)

        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch_num in range(epochs):

            # Set our model to training mode
            model.train()  # For transformers allow train.
            total_loss_train = 0

            for step, train_input in tqdm(enumerate(train_loader), total=len(train_loader)):
                train_label = train_input['label'].reshape(-1, 1).to(device)
                mask = train_input['mask'].to(device)
                input_id = train_input['ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                # print("Output shape", output.shape), print("Train shape", train_label.shape)

                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
                scheduler.step()

            total_loss_val = 0

            with torch.no_grad():

                for step, val_input in tqdm(enumerate(val_loader), total=len(val_loader)):
                    val_label = val_input['label'].reshape(-1, 1).to(device)
                    mask = val_input['mask'].to(device)
                    input_id = val_input['ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()

            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_ids): .6f} '
                f'Val Loss: {total_loss_val / len(val_ids): .6f}')

        # Saving the model
        save_model(model, folds_path, fold_i=fold)


def save_model(model, path, fold_i=0):
    if not os.path.exists(path):
        os.makedirs(path)

    output_dir = os.path.join(path, f'model_fold_{fold_i}')
    torch.save(model.state_dict(), output_dir)


def inference(model_paths, dataloader, device):
    final_preds = []
    for i, path in enumerate(model_paths):
        model = JigsawModel(CONFIG_TRAIN['model_name'], CONFIG_TRAIN['num_classes'])
        model.to(device)
        model.load_state_dict(torch.load(path))

        print(f"Getting predictions for model {i + 1}")
        preds = valid_fn(model, dataloader, device)
        final_preds.append(preds)

    final_preds = np.array(final_preds)
    final_preds = np.mean(final_preds, axis=0)
    return final_preds


if __name__ == '__main__':

    set_seed(CONFIG_TRAIN['seed'])

    REMOTE = True
    TRAIN, VALIDATE = False, True

    if REMOTE:
        path = '/home/daca/kaggle_jigsaw/jigsaw_comments/data/'
    else:
        path = 'data/'

    if TRAIN:
        ruddit_data, toxic_data, toxic_multiling_data = \
            pd.read_csv(os.path.join(path, 'ruddit_with_text.csv')), \
            pd.read_csv(os.path.join(path, 'train.csv')), \
            pd.read_csv(os.path.join(path, 'jigsaw-toxic-comment-train.csv'))

        # Train process
        toxic_data, ruddit_data, toxic_multiling_data = prepare_datasets(ruddit_data, toxic_data, toxic_multiling_data)
        # Combine 3 datasets
        dataset = pd.concat([toxic_data, ruddit_data, toxic_multiling_data], ignore_index=True)
        # Combine data and prepare the JigsawDataset
        # train_dataset = JigsawDataset(dataset, CONFIG_TRAIN['tokenizer'], max_length=CONFIG_TRAIN['max_length'])
        # train_loader = DataLoader(train_dataset, batch_size=CONFIG_TRAIN['test_batch_size'],
        #                           num_workers=2, shuffle=False, pin_memory=True)
        train_fn(dataset, CONFIG_TRAIN['device'], learning_rate=1e-6, epochs=5, k_folds=3)

    if VALIDATE:
        # Test and prediction object
        df_val = pd.read_csv("data/validation_data.csv")

        MODEL_PATHS = [os.path.join(folds_path, item) for item in os.listdir(folds_path)]
        CONFIG_TEST["tokenizer"] = AutoTokenizer.from_pretrained(CONFIG_TEST['model_name'])

        data_val_less = JigsawDataset(df_val['less_toxic'], CONFIG_TEST['tokenizer'], max_length=CONFIG_TEST['max_length'])
        data_val_less_loader = DataLoader(data_val_less, batch_size=CONFIG_TEST['test_batch_size'], shuffle=False)
        data_val_more = JigsawDataset(df_val['more_toxic'], CONFIG_TEST['tokenizer'], max_length=CONFIG_TEST['max_length'])
        data_val_more_loader = DataLoader(data_val_more, batch_size=CONFIG_TEST['test_batch_size'], shuffle=False)

        preds1 = inference(MODEL_PATHS, data_val_less_loader, CONFIG_TEST['device'])
        preds2 = inference(MODEL_PATHS, data_val_more_loader, CONFIG_TEST['device'])

        print(f'Validation Accuracy is { np.round((preds1 < preds2).mean() * 100,2)}')
