from pathlib import Path
from omegaconf import OmegaConf

# LOAD CONFIG-------------------------------------------------------------------------------------------------------------
config_path = '/home/gdli7/IBench/config/config.yaml'
if not Path(config_path).exists():
    raise FileExistsError(f'{config_path} does not exist!')
config = OmegaConf.load(config_path)
