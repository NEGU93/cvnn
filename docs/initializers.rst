Initializers
============

**Motivation**

For complex-valued glorot [GLOROT-2010]_ or He [HE-2015]_ initialization, one can not simply initialize real and complex part separately or one will not have the variance restrictions presented on the papers and therefore will lose its good properties.
The theory on how to implement the initialization can be found in [TRABELESI-2017]_ section 3.6.

.. toctree::
	:maxdepth: 2

    initializers/glorot_uniform
    initializers/glorot_normal
    initializers/he_normal
    initializers/he_uniform

.. [GLOROT-2010] Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural networks." Proceedings of the thirteenth international conference on artificial intelligence and statistics. 2010.

.. [HE-2015] He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet classification." Proceedings of the IEEE international conference on computer vision. 2015.

.. [TRABELESI-2017] Trabelsi, Chiheb et al. "Deep Complex Networks" arXiv:1705.09792 [cs]. 2017.