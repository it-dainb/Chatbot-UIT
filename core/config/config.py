from configparser import ConfigParser

_verbose: bool = False
_config: ConfigParser = None

def load_config(path) -> None:
    global _config
    _config = ConfigParser()
    _config.read(path)
    
    _verbose = get_config("Dev", "verbose")
    

def get_config(section, option):
    if _config is None:
        load_config("config.cfg")

    config = _config.get(section, option)
    config = str(config)

    if config.isnumeric():
        return int(config)

    if all([char.isnumeric() for char in config.split(".")]):
        return float(config)

    if config.lower() == "true":
        return True
    elif config.lower() == "false":
        return False
        
    return config

def is_verbose() -> bool:
    return _verbose