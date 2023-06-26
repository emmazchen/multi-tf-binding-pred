# for command line args and json config file parsing
import sys
import json
# for linking .py files
from model.classification_transformer import *
from lightning_modules import *
from torch.utils.data import DataLoader


# for logging results
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.dataset_classes import *
from utils.padding_collate_fn import *
#from utils.batch_sampler import *


configfile = f"configs/ablate_self_config.json" 
with open(configfile) as stream:
    config = json.load(stream)

model_config = config['model_config']
loss_config = config['loss_config']
optim_config = config['optim_config']
trainer_config = config['trainer_config']

#### dataset #########
dataset = "3_without0_nl"
seen="seen" #seen or unseen 
######################

train_raw = torch.load(f"data/{dataset}/train_set_{dataset}.pt")
val_raw = torch.load(f"data/{dataset}/val_set_{seen}tf_{dataset}.pt")
test_raw = torch.load(f"data/{dataset}/test_set_{seen}tf_{dataset}.pt")
# train_set is list of tuples => [(prot, dna, prob), (prot, dna, prob)...]

train_set = PretrainedPairDataset(train_raw)
val_set = PretrainedPairDataset(val_raw)
test_set = PretrainedPairDataset(test_raw)

# load data 
train_dl = DataLoader(train_set, collate_fn=padding_collate_pretrainedpair, batch_size=config['batch_size'], shuffle=True)
val_dl = DataLoader(val_set, collate_fn=padding_collate_pretrainedpair, batch_size=config['batch_size'], shuffle=True) 
test_dl = DataLoader(test_set, collate_fn=padding_collate_pretrainedpair, batch_size=config['batch_size'], shuffle=True) 

# instance model
model = eval(model_config['model_name'])(model_config['model_kwargs'])

# instance litmodelwrapper
litmodel = LitModelWrapper(model=model, loss_config=loss_config, optim_config=optim_config)

# instance wandb logger
plg= WandbLogger(project = config['wandb_project'],
                 entity = 'emmazchen', 
                 config=config) ## include run config so it gets logged to wandb 
plg.watch(litmodel) ## this logs the gradients for model 

## add the logger object to the training config portion of the run config 
trainer_config['logger'] = plg

## set to save every checkpoint (lightning saves the best checkpoint of model by default)
checkpoint_cb = ModelCheckpoint(save_top_k=-1, every_n_epochs = None, every_n_train_steps = None, train_time_interval = None)
trainer_config['callbacks'] = [checkpoint_cb]
trainer = pl.Trainer(**trainer_config)

# dry run lets you check if everythign can be loaded properly 
if config['dryrun']:
    print("Successfully loaded everything. Quitting")
    sys.exit()

# train
trainer.fit(litmodel, train_dataloaders = train_dl, val_dataloaders=val_dl)

# test
trainer.test(litmodel, dataloaders = test_dl)

# visualize
targets = torch.cat([x for x in litmodel.test_targets], dim=0).cpu().numpy()
preds = torch.cat([x for x in  litmodel.test_preds], dim=0).cpu().numpy()

torch.save(targets,"test_results/targets_ablateself.pt")
torch.save(preds, "test_results/preds_ablateself.pt")
