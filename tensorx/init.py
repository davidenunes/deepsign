# Copyright 2017 Davide Nunes

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================================================================================================================

import numpy as np
import tensorflow as tf
from functools import partial


def glorot_init(shape, dtype=tf.float32):
    """Normalised Weight Initialisation ("Xavier Init").

            Keeps the scale of the gradients roughly the same in all layers:
            (Glorot and Yoshua Bengio (2010),"Understanding the difficulty
            of training deep feedforward neural networks".

            The idea is to try and mitigate problems like:
                vanishing gradient: the gradient decreases exponentially
                    (multiplication throughout each layer) and the front
                    layers train very slowly. (Affects deep networks and
                    recurrent networks --more multiplications);

                exploring gradient: when we use activation functions
                    whose derivative can take larger values,gradients
                    can grow exponentially.

            Use Case:
                When we assume activations are linear - not true for ReLU and PReLU units.

            Args:
                shape:     [fan_in, fan_out]
                dtype:     tensorflow data type

            Returns:
                A norma initializer for weight matrices.
        """
    [fan_in, fan_out] = shape
    low = -np.sqrt(6.0 / (fan_in + fan_out))
    high = np.sqrt(6.0 / (fan_in + fan_out))

    return tf.random_uniform((fan_in, fan_out),
                             minval=low,
                             maxval=high,
                             dtype=dtype)


def relu_weight_init(shape, dtype=tf.float32):
    """Initialises the weights with a Gaussian distribution with:
        mu: 0
        sigma: sqrt(2/fan_in)

        Liner Neuron Assumption: immediately after initialisation, the parts of tanh and sigm that are being explored
        are close to zero, the gradient is close to one. This doesn't hold for rectifying non-linearities for this case,
        this initialisation comes from (He, Rang, Zhen and Sun 2015)"Delving Deep into Rectifiers:Surpassing Human-Level
        Performance on ImageNet Classification".

    Use Case:
        with deep networks that use ReLU and PReLU units.

    Args:
        shape: [fan_in, fan_out]
        :param shape: shape of the tensor to be generated
        :param dtype: tensorflow data type
    """
    [fan_in, fan_out] = shape
    mu = 0
    sigma = np.sqrt(2.0 / fan_in)
    return tf.random_normal((fan_in, fan_out),
                            mean=mu,
                            stddev=sigma,
                            dtype=dtype)


def random_uniform_init(shape, dtype=tf.float32):
    """
    
    :param shape: shape of the tensor to be generated 
    :param dtype: tensorflow data type
    :return: a new tensor with the given shape with entries initalised to random uniform values between -1 and 1
    """
    return tf.random_uniform(shape, minval=-1, maxval=1, dtype=dtype)
