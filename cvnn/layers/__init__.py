# https://stackoverflow.com/questions/24100558/how-can-i-split-a-module-into-multiple-files-without-breaking-a-backwards-compa/24100645
from cvnn.layers.pooling import ComplexMaxPooling2D, ComplexAvgPooling2D
from cvnn.layers.convolutional import ComplexConv2D, ComplexConv1D, ComplexConv3D
from cvnn.layers.core import ComplexInput, ComplexDense, ComplexFlatten, ComplexDropout, complex_input


__author__ = 'J. Agustin BARRACHINA'
__copyright__ = 'Copyright 2020, {project_name}'
__credits__ = ['{credit_list}']
__license__ = '{license}'
__version__ = '1.0.6'
__maintainer__ = 'J. Agustin BARRACHINA'
__email__ = 'joseagustin.barra@gmail.com; jose-agustin.barrachina@centralesupelec.fr'
__status__ = '{dev_status}'
