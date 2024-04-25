import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler

class SequentialDataset(Dataset):
    def __init__(self, data_config):
        """
        Args:
            data_config (dict): Data Loading config
        """
        self.data = pd.read_csv(data_config['dataset_dir'])
        self.sequence_length = data_config['seq_len']
        self.event_ids = list(self.data.groupby('event_id').groups.keys())
        
        
    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, idx):
        
        sequence = self.data[self.data['event_id'] == idx]
        X = torch.tensor(sequence.drop(columns = ['risk','event_id'], axis=1).values)
        y = torch.tensor([sequence['risk'].values[-1]])

        # Pad the sequence if it's shorter than the required length
        if len(X) < self.sequence_length:
            X = torch.cat([X, torch.zeros(self.sequence_length - len(X), X.shape[1])], dim=0)
            
        # Use the latest values of sequence if the sequence longer than the required sequence_length
        elif len(X) > self.sequence_length:
            X = X[-self.sequence_length:]

        return X.float(), y.float()

class SequentialDataModule(pl.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        
        self.data_config = data_config
        self.dataset = SequentialDataset(self.data_config)

        self.data_split()

        
    def data_split(self):

        # Split the data into train, validation, and test sets
        train_val_event_ids, test_event_ids = train_test_split(self.dataset.event_ids, test_size=self.data_config['test_split'], random_state=self.data_config['seed_value'])
        train_event_ids, val_event_ids = train_test_split(train_val_event_ids, test_size=self.data_config['validation_split'], random_state=self.data_config['seed_value'])

        # Creating data samplers and loaders:
        self.train_sampler = SubsetRandomSampler(train_event_ids)
        self.val_sampler = SubsetRandomSampler(val_event_ids)
        self.test_sampler = SubsetRandomSampler(test_event_ids[:10])

        
    def __len__(self):
        return len(self.dataset)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.data_config['batch_size'], sampler=self.train_sampler)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.data_config['batch_size'], sampler=self.val_sampler)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.data_config['batch_size'], sampler=self.test_sampler)


if __name__ == '__main__':
    import yaml
    config_dir = 'config.yaml'
    with open(config_dir) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
    data_module = SequentialDataModule(config['data_config'])
    
    for x, y in data_module.train_dataloader():
        print(x,"\n\n",y)
        break

    for x, y in data_module.val_dataloader():
        break
    
    print("\nShape of input:", x.shape, "\nShape of output:",y.shape)
    #print("\nSample input:",x[0],"\nSample output:",y[0])
    print("\nLength of dataset:",len(data_module))