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

    return _config.get(section, option)

def is_verbose() -> bool:
    return _verbose