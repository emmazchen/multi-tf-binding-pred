#%%
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


configfile = f"configs/pretrained_config.json" 
with open(configfile) as stream:
    config = json.load(stream)

model_config = config['model_config']
loss_config = config['loss_config']
optim_config = config['optim_config']
trainer_config = config['trainer_config']

#### dataset #########
dataset = "3_without0_nl"
seen="unseen" #unseen or seen
######################

# only need test data
test_raw = torch.load(f"data/{dataset}/test_set_{seen}tf_{dataset}.pt")
test_set = PretrainedPairDataset(test_raw)
test_dl = DataLoader(test_set, collate_fn=padding_collate_pretrainedpair, batch_size=config['batch_size'], shuffle=True) 

# instance model
model = eval(model_config['model_name'])(model_config['model_kwargs'])
litmodel = LitModelWrapper(model=model, loss_config=loss_config, optim_config=optim_config)
#litmodel = LitModelWrapper.load_from_checkpoint("multi-tf-pretrained-seentf/lr=1e-4,batchsz=5/checkpoints/epoch=9-step=16170.ckpt")

# instance litmodelwrapper
trainer = pl.Trainer(**trainer_config)

# test
trainer.test(litmodel, dataloaders = test_dl, ckpt_path="multi-tf-pretrained-seentf/lr=1e-4,batchsz=5,epoch=20/checkpoints/epoch=19-step=32340.ckpt")

targets = torch.cat([x for x in litmodel.test_targets], dim=0).cpu().numpy()
preds = torch.cat([x for x in  litmodel.test_preds], dim=0).cpu().numpy()

# save the pred vs targets data to be visualized
torch.save(targets,"test_results/targets_epoch=20_unseentf.pt")
torch.save(preds, "test_results/preds_epoch=20_unseentf.pt")


# %%
