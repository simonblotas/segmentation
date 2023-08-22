"""
segmentation base module.

This is the principal module of the segmentation project.

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
def compute_v_value(signal: jnp.ndarray, projection: jnp.ndarray, segmentation_size: int, beta: float) -> float:
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
        compute_v_value(transformed_signal, true_projection, true_segmentation_size, beta) -
        compute_v_value(transformed_signal, pred_projection, pred_segmentation_size, beta)
    ) / true_segmentation_size

    return loss_value



@jax.jit
def main_loss(params: dict, signal: jnp.ndarray, true_segmentation: jnp.ndarray, lmbda: float) -> float:
    '''
    Computes the main loss function, which includes a forward pass, segmentation loss, and regularization.

    Parameters:
    params (dict): A dictionary containing model parameters, including 'beta' and 'omega_weights'.
    signal (jnp.ndarray): The input signal.
    true_segmentation (jnp.ndarray): The true segmentation points.
    lmbda (float): The regularization parameter.

    Returns:
    float: The computed main loss value.
    '''

    # Perform a forward pass to transform the signal using the model parameters
    # The transformation applied depends on the chosen transformation in the forward_pass function
    transformed_signal = forward_pass(params, signal)
    
    # Calculate the segmentation loss using the transformed signal, penalty parameter, and true segmentation
    segmentation_loss = loss(transformed_signal, params['beta'], true_segmentation)
    
    # Calculate the regularization term based on the absolute values of the 'omega_weights'
    regularization_term = lmbda * jnp.sum(jnp.abs(params['omega_weights']))
    
    # Compute the total main loss by combining the segmentation loss and regularization term
    main_loss_value = segmentation_loss + regularization_term
    
    return main_loss_value

# Bacthed version of jax's value_and_grad function for main_loss
batched_value_and_grad = jax.jit(vmap(value_and_grad(main_loss, argnums=0, allow_int=True), in_axes=(None,0,0,None), out_axes=0 ))


def final_loss_and_grad(params: dict, signals: jnp.ndarray, true_segmentation: jnp.ndarray) -> Tuple[float, dict]:
    '''
    Compute the final loss value and aggregated gradients for a batch of signals and true segmentations.

    Parameters:
    params (dict): A dictionary containing model parameters.
    signals (jnp.ndarray): An array of input signals.
    true_segmentation (jnp.ndarray): An array of true segmentation points.

    Returns:
    Tuple[float, dict]: A tuple containing the final loss value and aggregated gradients.
    '''

    # Compute batched loss values and gradients
    losses, grads = batched_value_and_grad(params, signals, true_segmentation)
    
    # Compute the final loss as the sum of batched losses
    final_loss = jnp.sum(losses)
    
    # Aggregate gradients for each parameter
    for name, value in grads.items():
        grads[name] = jnp.sum(value, axis=0)
        
    return final_loss, grads
