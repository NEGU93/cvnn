from typing import Union, Type, List, Set, Optional, Tuple, Callable
from tensorflow import Tensor, data, dtypes
from tensorflow.keras.losses import Loss
from cvnn.layers import ComplexLayer
from numpy import ndarray, dtype
from cvnn.optimizers import Optimizer

# Typing
t_Dtype = Union[dtypes.DType, dtype]
t_layers_shape = Union[ndarray, List[ComplexLayer], Set[ComplexLayer]]
t_loss_fun = Type[Loss]
t_List = Union[ndarray, list]
t_input_features = Union[t_List, Tensor, data.Dataset]
t_labels = Union[t_List, Tensor]
t_optimizer = Union[str, Optimizer]
t_verbose = Union[str, int, bool]
