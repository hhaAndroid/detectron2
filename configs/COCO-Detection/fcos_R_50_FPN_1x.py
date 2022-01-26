from ..common.optim import SGD as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
from ..common.data.coco import dataloader
from ..common.models.fcos import model
from ..common.train import train

dataloader.train.mapper.use_instance_mask = False
optimizer.lr = 0.01

model.backbone.bottom_up.freeze_at = 2
train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"

# ceph
file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        'datasets/': 's3://openmmlab/datasets/detection/',
        '.datasets/': 's3://openmmlab/datasets/detection/'
    }))

# If you don’t need ceph, you can directly comment the following code or
# set file_client_args to None
dataloader.train.mapper.file_client_args = file_client_args
dataloader.test.mapper.file_client_args = file_client_args
