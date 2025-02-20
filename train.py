# from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from colorization_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from colorization_dataset import ColorizationDataset

sd_version = "21"

# Configs
# resume_path = f'./models/control_sd{sd_version}_ini.ckpt'
resume_path = "weights/controlnet/v2-1_colorization.pt"
batch_size = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(f'./models/cldm_v{sd_version}.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = ColorizationDataset("data/colorization/")
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True, pin_memory=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=16, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)
