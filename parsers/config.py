import yaml
from easydict import EasyDict as edict


def get_config(config, seed):
    config_dir = f'./model/config/' + config + '.yaml'
    config = edict(yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader))
    config.seed = seed
    print(seed)
    return config
