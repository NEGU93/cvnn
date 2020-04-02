import logging
import colorlog

# How to comment script header
# https://medium.com/@rukavina.andrei/how-to-write-a-python-script-header-51d3cec13731
__author__ = 'J. Agustin BARRACHINA'
__copyright__ = 'Copyright 2020, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '0.2.56'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
__status__ = '{dev_status}'

STRING_FORMATTER = "%(asctime)s — %(levelname)s:%(name)s — %(message)s"

FORMATTER = logging.Formatter(STRING_FORMATTER)

file_handler = logging.FileHandler("./log/logs.log")
file_handler.setFormatter(FORMATTER)

# https://github.com/borntyping/python-colorlog
# https://stackoverflow.com/a/23964880/5931672
console_handler = colorlog.StreamHandler()
console_handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s' + STRING_FORMATTER))

logger = colorlog.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)
logger.addHandler(file_handler)



