import colorlog
from colorlog import ColoredFormatter

formatter = ColoredFormatter(
    fmt='{log_color}{asctime} [ {levelname:<10}] {name}: {reset} {blue}{message}',
	datefmt='%H:%M:%S',
	reset=True,
	log_colors={
		'DEBUG':    'cyan',
		'INFO':     'green',
		'WARNING':  'yellow',
		'ERROR':    'red',
		'CRITICAL': 'red,bg_white',
	},
	secondary_log_colors={},
	style='{'
)

handler = colorlog.StreamHandler()
handler.setFormatter(formatter)

colorlog.getLogger().setLevel(colorlog.INFO)
colorlog.getLogger().addHandler(handler)

logger = colorlog.getLogger("CHATBOT_UIT")
logger.setLevel(colorlog.DEBUG)