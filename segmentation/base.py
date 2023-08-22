"""
segmentation base module.

This is the principal module of the segmentation project.
here you put your main classes and objects.

Be creative! do whatever you want!

If you want to replace this with a Flask application run:

    $ make init

and then choose `flask` as template.
"""


# Imports :
import jax
import jax.lax as lax
from flax import linen as nn
import matplotlib.pyplot as plt
from jax import random
import numpy as np
import flax
import optax
import ruptures as rpt
import jax.numpy as jnp
from jax import grad, jit
import pandas as pd
from typing import Tuple
import copy
import jax
from jax import jit, lax
from jax import numpy as jnp
from jax.ops import segment_sum
from jax.tree_util import Partial
from jax import grad, jit, vmap, value_and_grad
import sys
import time

from segmentation.utils import get_optimal_projection, segmentation_to_projection

@jit
def v(signal: jnp.ndarray, projection: jnp.ndarray, segmentation_size: int, beta: float) -> float:
    '''
    Computes the value of the function V for a given signal, projection, segmentation size, and penalty parameter.

    Parameters:
    signal (jnp.ndarray): The signal.
    projection (jnp.ndarray): The projection of the signal.
    segmentation_size (int): The size of the segmentation.
    beta (float): The penalty parameter.

    Returns:
    float: The computed value of the function V.
    '''

    return ((signal - projection) ** 2).sum() + jnp.exp(beta) * segmentation_size

@jit
def loss(transformed_signal: jnp.ndarray, beta: float, true_segmentation: jnp.ndarray) -> float:
    '''
    Computes the loss function for a given transformed signal, penalty 
    parameter, and true segmentation.

    Parameters:
    transformed_signal (jnp.ndarray): The transformed signal.
    beta (float): The penalty parameter.
    true_segmentation (jnp.ndarray): The true segmentation points.

    Returns:
    float: The computed loss value.
    '''

    # Calculate the projection and segment IDs using a prediction function
    pred_projection, pred_segmentation_size, segment_ids_pred = get_optimal_projection(transformed_signal, penalty=jnp.exp(beta))

    # Calculate the true projection and segmentation size
    true_projection = segmentation_to_projection(transformed_signal, true_segmentation)
    true_segmentation_size = true_segmentation[-1] + 1

    # Calculate the loss based on a difference in V values
    loss_value = jnp.sum(
        v(transformed_signal, true_projection, true_segmentation_size, beta) -
        v(transformed_signal, pred_projection, pred_segmentation_size, beta)
    ) / true_segmentation_size

    return loss_value




