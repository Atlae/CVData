# from trainer import ClassificationTrainer
import torch
from src import *

if __name__ == '__main__':
    image_channels: int = 3
    mask_channels: int = 1
    batch_size: int = 4
    learning_rate: float = 2.5e-4
    epochs: int = 4

    root = '/workspaces/CVData/datasets/Tomato'
    alias = Alias([
        Generic("{}", DataTypes.IMAGE_NAME),
        Generic("{}_{}", DataTypes.CLASS_NAME, DataTypes.GENERIC)
    ])
    form = {
        Generic("{}", DataTypes.IMAGE_SET_NAME): {
            "images": {
                File("{}.jpg", DataTypes.IMAGE_NAME): Image()
            },
            "labels": {
                File("{}.txt", DataTypes.IMAGE_NAME): TXTFile(
                    
                )
            }
        },
    }
    cvdata = CVData(root, form)
    cvdata.cleanup()
    cvdata.split_image_set('trainval', ('train', 0.8), ('val', 0.2), inplace = True, seed = 0)
    trainloader = cvdata.get_dataloader('classification', 'train', batch_size=batch_size, transforms=CVData.CLASSIFICATION_TRANSFORMS)
    valloader = cvdata.get_dataloader('classification', 'val', batch_size=batch_size, transforms=CVData.CLASSIFICATION_TRANSFORMS)
    testloader = cvdata.get_dataloader('classification', 'test', batch_size=batch_size, transforms=CVData.CLASSIFICATION_TRANSFORMS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        'device': device,
        'model_args': {
            'name': 'vgg16',
            'weight_type': None
        },
        'optimizer_args': {
            'name': 'SGD',
            'lr': learning_rate
        },
        'dataloaders': {
            'train': trainloader,
            'test': testloader,
            'val': valloader,
            'classes': cvdata.class_to_idx
        },
        'checkpointing': True,
        'num_epochs': 25
    }
    # trainer = ClassificationTrainer.from_config(config)
    # trainer.do_training('run_1_cls')
