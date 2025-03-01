from config.mappo_config import CommunicationType, MAPPOConfig


def config_to_dict(config):
    def is_primitive_type(obj):
        return isinstance(
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
    config = MAPPOConfig.create()

    def _dict_to_config(_dictionary, config_node):
        for key, value in _dictionary.items():
            if isinstance(value, dict):
                config_node = config_node._replace(
                    **{key: _dict_to_config(value, getattr(config_node, key))}
                )
            else:
                config_node = config_node._replace(**{key: value})
        return config_node

    config = _dict_to_config(dictionary, config)

    return config
