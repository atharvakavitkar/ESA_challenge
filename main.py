import os
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

# User-defined modules
from models.GRU import GRUModel
from data_pipeline.train_data import SequentialDataModule
from trainer.trainer import RiskTrainer

def get_config(config_dir):
    """
    Load the configuration from a YAML file.

    Args:
        config_dir (str): The directory where the configuration file is located.

    Returns:
        dict: The configuration dictionary.
    """
    config_file = os.path.join(config_dir, 'config.yaml')
    with open(config_file) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def train(config):
    """
    Train the GRU model using the provided configuration.

    Args:
        config (dict): The configuration dictionary.
    """
    callbacks = []
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=2,
                                       monitor='val_loss',
                                       filename='{epoch}-{val_loss:.4f}')
    callbacks.append(model_checkpoint)

    model = GRUModel(**config['model_config'])
    print(model)

    regressor = RiskTrainer(model=model,
                           lr=config['train_config']['learning_rate'])

    trainer = pl.Trainer(max_epochs=config['train_config']['epochs'],
                         logger=CSV_logger,
                         callbacks=callbacks)

    trainer.fit(regressor, datamodule=data)


def test(config):
    """
    Test the GRU model using the provided configuration.

    Args:
        config (dict): The configuration dictionary.
    """
    model = GRUModel(**config['model_config'])
    pretrained_model = RiskTrainer.load_from_checkpoint(config['inference_config']['pretrained_ckpt'],
                                                        model=model)
    trainer = pl.Trainer(logger=False)

    test_dataloader = SequentialDataModule(config['data_config']).test_dataloader()

    predictions = trainer.predict(model=pretrained_model,
                                  dataloaders=test_dataloader)

    print(predictions)


if __name__ == "__main__":
    # Load the configuration
    config_dir = ''
    config = get_config(config_dir)
    print(yaml.dump(config))

    # Set up the logger
    CSV_logger = CSVLogger(save_dir=config['logger_config']['log_dir'])
    CSV_logger.log_hyperparams(config)

    # Set the seed for reproducibility
    pl.seed_everything(config['data_config']['seed_value'])

    # Check the availability of CUDA
    print('\nIs CUDA available to PyTorch?:', torch.cuda.is_available())
    print('Number of GPUs visible to PyTorch: ', torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the data
    data = SequentialDataModule(config['data_config'])

    # Train or test the model based on the configuration
    if config['train_config']['train_mode']:
        train(config)
    else:
        test(config)