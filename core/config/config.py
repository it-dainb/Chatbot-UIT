from configparser import ConfigParser

_config: ConfigParser = None

def load_config(path) -> None:
    """
    Load configuration settings from a file.

    Args:
        path (str): The path to the configuration file.

    Returns:
        None
    """
    global _config
    _config = ConfigParser()
    _config.read(path)    

def get_config(section, option):
    """
    Retrieve a configuration value from the loaded configuration.

    Args:
        section (str): The section name in the configuration.
        option (str): The option name within the specified section.

    Returns:
        Union[int, float, bool, str]: The configuration value.
    """
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