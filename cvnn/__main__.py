from cvnn import cli
import sys
import logging

cli.cli()

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(FORMATTER)
logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)
