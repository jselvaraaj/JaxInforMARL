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


def dict_to_config(dictionary) -> MAPPOConfig:
    config = MAPPOConfig()

    def _dict_to_config(_dictionary):
        nonlocal config
        for key, value in _dictionary.items():
            if isinstance(value, dict):
                config = config._replace(**{key: dict_to_config(value)})
            else:
                config = config._replace(**{key: value})

    _dict_to_config(dictionary)

    return config
