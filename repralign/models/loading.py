import timm
from mmaction.utils import register_all_modules

register_all_modules()
from mmaction.registry import MODELS as MMA_MODELS
from mmengine import Config
from mmengine.runner import load_checkpoint


def timm_loader(model_name):
    try:
        model = timm.create_model(model_name, pretrained=True, features_only=True)
    except:  # if features_only is not supported
        model = timm.create_model(model_name, pretrained=True)
    model.eval()
    return model


def mmaction_loader(path_to_config, path_to_checkpoint):
    cfg = Config.fromfile(path_to_config)
    model = MMA_MODELS.build(cfg.model)
    model.eval()
    load_checkpoint(model, path_to_checkpoint)
    return model
