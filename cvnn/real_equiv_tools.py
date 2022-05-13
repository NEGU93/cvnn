import sys
import numpy as np
from tensorflow.keras import Sequential
from pdb import set_trace
from cvnn import logger
import cvnn.layers as layers
from cvnn.layers.core import ComplexLayer
from typing import Type, List
from typing import Optional

EQUIV_TECHNIQUES = {
    "np", "alternate_tp", "ratio_tp", "none"
}


def get_real_equivalent_multiplier(layers_shape, classifier, equiv_technique, bias_adjust: bool = False):
    """
    Returns an array (output_multiplier) of size `self.shape` (number of hidden layers + output layer)
        one must multiply the real valued equivalent layer
    In other words, the real valued equivalent layer 'i' will have:
        neurons_real_valued_layer[i] = output_multiplier[i] * neurons_complex_valued_layer[i]
    :param layers_shape:
    :param classifier: Boolean (default = True) weather the model's task is to classify (True) or
        a regression task (False)
    :param equiv_technique: Used to define the strategy of the capacity equivalent model.
        This parameter is ignored if capacity_equivalent=False
        - 'np': double all layer size (except the last one if classifier=True)
        - 'ratio': neurons_real_valued_layer[i] = r * neurons_complex_valued_layer[i], 'r' constant for all 'i'
        - 'alternate': Method described in https://arxiv.org/abs/1811.12351 where one alternates between
                multiplying by 2 or 1. Special case in the middle is treated as a compromise between the two.
    :return: output_multiplier
    """
    dense_layers = [d for d in layers_shape if isinstance(d, layers.ComplexDense)]      # Keep only dense layers
    return get_real_equivalent_multiplier_from_shape(_parse_sizes(dense_layers), classifier=classifier,
                                                     equiv_technique=equiv_technique, bias_adjust=bias_adjust)


def get_real_equivalent_multiplier_from_shape(layers_shape: List[int], equiv_technique: str,
                                              classifier: bool = True,  bias_adjust: bool = False):
    equiv_technique = equiv_technique.lower()
    if equiv_technique not in EQUIV_TECHNIQUES:
        raise ValueError(f"Unknown equiv_technique {equiv_technique}")
    if equiv_technique == "alternate_tp":
        output_multiplier = _get_alternate_capacity_equivalent(layers_shape, classifier)
    elif equiv_technique == "ratio_tp":
        output_multiplier = _get_ratio_capacity_equivalent(layers_shape, classifier,
                                                           bias_adjust=bias_adjust)
    elif equiv_technique == "np":
        output_multiplier = 2 * np.ones(len(layers_shape)-1).astype(int)
        if classifier:
            output_multiplier[-1] = 1
    elif equiv_technique == "none":
        output_multiplier = np.ones(len(layers_shape) - 1).astype(int)
    else:
        raise ValueError(f"Unknown equiv_technique {equiv_technique} but listed on {EQUIV_TECHNIQUES}.")
    return output_multiplier


def get_real_equivalent(complex_model: Type[Sequential], classifier: bool = True, capacity_equivalent: bool = True,
                        equiv_technique: str = 'ratio', name: Optional[str] = None):
    assert isinstance(complex_model, Sequential), "Sorry, only sequential models supported for the moment"
    equiv_technique = equiv_technique.lower()
    if equiv_technique not in {"ratio", "alternate"}:
        logger.error("Invalid `equivalent_technique` argument: " + equiv_technique)
        sys.exit(-1)
    # assert len(self.shape) != 0
    real_input_shape = [inp for inp in complex_model.layers[0].input_shape if inp is not None]
    real_input_shape[-1] = real_input_shape[-1]*2
    real_shape = [layers.ComplexInput(input_shape=real_input_shape,
                                      dtype=complex_model.layers[0].input.dtype.real_dtype)]
    output_multiplier = get_real_equivalent_multiplier(complex_model.layers,
                                                       classifier, capacity_equivalent, equiv_technique)
    counter = 0
    for layer in complex_model.layers:
        if isinstance(layer, ComplexLayer):
            if isinstance(layer, layers.ComplexDense):  # TODO: Check if I can do this with kargs or sth
                real_shape.append(layer.get_real_equivalent(
                    output_multiplier=output_multiplier[counter]))
                counter += 1
            else:
                real_shape.append(layer.get_real_equivalent())
        else:
            sys.exit("Layer " + str(layer) + " unknown")
    assert counter == len(output_multiplier)
    if name is None:
        name = f"{complex_model.name}_real_equiv"
    real_equiv = Sequential(real_shape, name=name)
    real_equiv.compile(optimizer=complex_model.optimizer.__class__(), loss=complex_model.loss,
                       metrics=['accuracy'])
    return real_equiv


def _parse_sizes(dense_layers):
    assert len(dense_layers[0].input_shape) == 2, "Possibly a bug of cvnn. Please report it to github issues"
    model_in_c = dense_layers[0].input_shape[-1]  # -1 not to take the None part
    model_out_c = dense_layers[-1].units
    x_c = [dense_layers[i].units for i in range(len(dense_layers[:-1]))]
    x_c.insert(0, model_in_c)
    x_c.append(model_out_c)
    return x_c


def _get_ratio_capacity_equivalent(layers_shape, classification: bool = True, bias_adjust: bool = True):
    """
    Generates output_multiplier keeping not only the same capacity but keeping a constant ratio between the
                                                                                                    model's layers
    This helps keeps the 'aspect' or shape of the model my making:
        neurons_real_layer_i = ratio * neurons_complex_layer_i
    :param layers_shape:
    :param classification: True (default) if the model is a classification model. False otherwise.
    :param bias_adjust: True (default) if taking into account the bias as a trainable parameter. If not it will
        only match the real valued parameters of the weights
    """
    p_c = 0
    for i in range(len(layers_shape[:-1])):
        p_c += 2 * layers_shape[i] * layers_shape[i+1]
    model_in_c = layers_shape[0]
    model_out_c = layers_shape[-1]
    x_c = layers_shape[1:-1]
    if bias_adjust:
        p_c = p_c + 2 * np.sum(x_c) + 2 * model_out_c
    model_in_r = 2 * model_in_c
    model_out_r = model_out_c if classification else 2 * model_out_c
    # Quadratic equation
    if len(x_c) > 1:
        quadratic_c = float(-p_c)
        quadratic_b = float(model_in_r * x_c[0] + model_out_r * x_c[-1])
        if bias_adjust:
            quadratic_b = quadratic_b + np.sum(x_c) + model_out_c
        quadratic_a = float(np.sum([x_c[i] * x_c[i + 1] for i in range(len(x_c) - 1)]))
        # The result MUST be positive so I use the '+' solution
        ratio = (-quadratic_b + np.sqrt(quadratic_b ** 2 - 4 * quadratic_c * quadratic_a)) / (2 * quadratic_a)
        if not 1 <= ratio < 2:
            logger.error("Ratio {} has a weird value. This function must have a bug.".format(ratio))
    else:
        ratio = 2 * (model_in_c + model_out_c) / (model_in_r + model_out_r)
    return [ratio] * len(x_c) + [1 if classification else 2]


def _get_alternate_capacity_equivalent(layers_shape, classification: bool = True):
    """
    Generates output_multiplier using the alternate method described in https://arxiv.org/abs/1811.12351 which
        doubles or not the layer if it's neighbor was doubled or not (making the opposite).
    The code fills output_multiplier from both senses:
        output_multiplier = [ ... , .... ]
                             --->     <---
    If when both ends meet there's not a coincidence (example: [..., 1, 1, ...]) then
        the code will find a compromise between the two to keep the same real valued trainable parameters.
    """
    output_multiplier = np.zeros(len(layers_shape))
    output_multiplier[0] = 2                                # Sets input multiplier
    output_multiplier[-1] = 1 if classification else 2      # Output multiplier
    i: int = 1
    while i < (len(layers_shape) - i):     # Fill the hidden layers (from 1 to len()-1)
        output_multiplier[i] = 2 if output_multiplier[i - 1] == 1 else 1        # From beginning
        output_multiplier[-1 - i] = 2 if output_multiplier[-i] == 1 else 1      # From the end
        index_in_middle_with_diff_borders = i == len(layers_shape) - i - 1 and output_multiplier[i - 1] != output_multiplier[i + 1]
        subsequent_indexes_are_equal = i == len(layers_shape) - i and output_multiplier[i] == output_multiplier[i + 1]
        if index_in_middle_with_diff_borders or subsequent_indexes_are_equal:
            m_inf = layers_shape[i - 1]    # This is because dense_layers are len(output_multiplier) - 1
            m_sup = layers_shape[i + 1]
            if i == len(layers_shape) - i - 1:          # index_in_middle_with_diff_borders
                coef_sup = output_multiplier[i + 1]
                coef_inf = output_multiplier[i - 1]
            else:                                       # subsequent_indexes_are_equal
                coef_sup = output_multiplier[i + 1]
                coef_inf = output_multiplier[i]
            output_multiplier[i] = 2 * (m_inf + m_sup) / (coef_inf * m_inf + coef_sup * m_sup)
        i += 1
    return output_multiplier[1:]
