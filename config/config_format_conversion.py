import config.mappo_config as mappo_config_module
from config.mappo_config import MAPPOConfig, CommunicationType


def config_to_dict(config):
    is_primitive_type = lambda obj: isinstance(
        obj, (int, float, str, bool, type(None), CommunicationType, list)
    )
    return {
        attr: (
            getattr(config, attr)
            if is_primitive_type(getattr(config, attr))
            else config_to_dict(getattr(config, attr))
        )
        for attr in dir(config)
        if not callable(getattr(config, attr))
        and not (attr.startswith("__") or attr.startswith("_"))
    }


def dict_to_config(dictionary, name="MAPPOConfig") -> MAPPOConfig:
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = dict_to_config(value, key)
    return getattr(mappo_config_module, name)(**dictionary)
