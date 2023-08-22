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
from typing import Dict, List


from segmentation.utils import get_optimal_projection, segmentation_to_projection, find_change_indices

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


def final_loss_and_grad(params: dict, signals: jnp.ndarray, true_segmentation: jnp.ndarray, lmbda: float) -> Tuple[float, dict]:
    '''
    Compute the final loss value and aggregated gradients for a batch of signals and true segmentations.

    Parameters:
    params (dict): A dictionary containing model parameters.
    signals (jnp.ndarray): An array of input signals.
    true_segmentation (jnp.ndarray): An array of true segmentation points.
    lmbda (float): The regularization parameter.

    Returns:
    Tuple[float, dict]: A tuple containing the final loss value and aggregated gradients.
    '''

    # Compute batched loss values and gradients
    losses, grads = batched_value_and_grad(params, signals, true_segmentation, lmbda)
    
    # Compute the final loss as the sum of batched losses
    final_loss = jnp.sum(losses)
    
    # Aggregate gradients for each parameter
    for name, value in grads.items():
        grads[name] = jnp.sum(value, axis=0)
        
    return final_loss, grads



class Network:
    def __init__(self) -> None:
        pass # Implemented in child class


    def transform_signal(self, signal: jnp.ndarray) -> jnp.ndarray:
        pass # Implemented in child class
    # Make a batched version of the `forward_pass` function
    
    batch_transform_signal = vmap(transform_signal, in_axes=(None, 0), out_axes=0)

    def predict_segmentation(self, signal: jnp.ndarray) -> jnp.ndarray:
        """
        Predicts the segmentation of a given signal using the trained scaling network.

        Parameters:
        signal (jnp.ndarray): The input signal for segmentation prediction.

        Returns:
        jnp.ndarray: Predicted segmentation indices for the input signal.
        """
        # Transform the input signal using the network's transformation function
        transformed_signal = self.transform_signal(signal)
        
        # Convert the transformed signal to a numpy array for compatibility
        transformed_signal_array = np.array(transformed_signal)
        
        # Get the optimal projection and predicted segmentation using the transformed signal
        pred_projection, pred_segmentation_size, segment_ids_pred = get_optimal_projection(
            transformed_signal_array, penalty=jnp.exp(self.params['beta'])
        )
        
        # Find and return the predicted segmentation indices
        predicted_segmentation = find_change_indices(segment_ids_pred[0])
        return predicted_segmentation

    def train(self, num_epochs: int, test_signals: List[jnp.ndarray],
            strided_true_segmentations: jnp.ndarray, batch_size: int,
            ) -> Tuple[List[float], Dict, Dict]:
    # Training implementation 
    pass



class ScalingNetwork(Network):
    def __init__(self, n_dimensions: int, initial_beta: float = 1.):
        '''
        Initializes the ScalingNetwork instance.

        Parameters:
        n_dimensions (int): The number of dimensions.
        initial_beta (float, optional): The initial value of beta. Default is 1.
        '''
        
        super().__init__()
        self.params = self.params_init(n_dimensions, initial_beta)

    def params_init(self, n_dimensions: int, initial_beta: float) -> Dict[str, jnp.ndarray]:
        '''
        Initialize parameters for the scaling network.

        Parameters:
        n_dimensions (int): The number of dimensions.
        initial_beta (float): The initial value of beta.

        Returns:
        Dict[str, jnp.ndarray]: A dictionary containing initialized parameters.
        '''
        params = {}
        key = jax.random.PRNGKey(0)
        w_key, b_key = jax.random.split(key)
        omega_0_weights = jax.random.normal(w_key, (1, n_dimensions))
        omega_0_bias = jax.random.normal(b_key, (1, n_dimensions))
        params['omega_weights'] = omega_0_weights
        params['omega_bias'] = omega_0_bias
        params['beta'] = initial_beta
        return params

    @jit
    def transform_signal(self, signal: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the transformation of the scaling network to the given signal.

        Parameters:
        signal (jnp.ndarray): The input signal.

        Returns:
        jnp.ndarray: The scaled signal after applying the forward pass.
        """
        scaled_signal = signal * self.params['omega_weights'] + self.params['omega_bias']
        return scaled_signal

    






