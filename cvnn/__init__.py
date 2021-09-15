import logging
import colorlog
import re
import os
from cvnn.utils import create_folder
from tensorflow.keras.utils import get_custom_objects
from cvnn.activations import act_dispatcher
from cvnn.initializers import init_dispatcher

get_custom_objects().update(act_dispatcher)     # Makes my activation functions usable with a string
get_custom_objects().update(init_dispatcher)


def get_version() -> str:
    versionfile = os.path.split(os.path.realpath(__file__))[0] + "/_version.py"
    verstrline = open(versionfile, "rt").read()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        return mo.group(1)
    else:
        VSRE = r"\"version\": ['\"]([^'\"]*)['\"]"
        mo = re.search(VSRE, verstrline, re.M)
        if mo:
            return mo.group(1)
        else:
            raise RuntimeError("Unable to find version string in %s." % (versionfile,))


# How to comment script header
# https://medium.com/@rukavina.andrei/how-to-write-a-python-script-header-51d3cec13731
__author__ = 'J. Agustin BARRACHINA'
__copyright__ = 'Copyright 2020, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = get_version()
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
__status__ = '{dev_status}'

# logging.getLogger('tensorflow').disabled = True     # Removes https://github.com/tensorflow/tensorflow/issues/41557

STRING_FORMATTER = "%(asctime)s — %(levelname)s - %(module)s::%(funcName)s line %(lineno)s — %(message)s"

# file_handler = logging.FileHandler(create_folder("./log/logs/") / "logs.log")
# formatter = logging.Formatter(STRING_FORMATTER)
# file_handler.setFormatter(formatter)

# https://github.com/borntyping/python-colorlog
# https://stackoverflow.com/a/23964880/5931672
console_handler = colorlog.StreamHandler()
console_handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s' + STRING_FORMATTER))

logger = colorlog.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)
# logger.addHandler(file_handler)





