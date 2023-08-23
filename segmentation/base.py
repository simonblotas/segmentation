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
from typing import Dict, List, Generator, Tuple


from segmentation.utils import get_optimal_projection, segmentation_to_projection, find_change_indices, create_data_loader

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


class Network:
    def __init__(self) -> None:
        self.params = None
        pass  # Implemented in child class

    def transform_signal(self, signal: jnp.ndarray) -> jnp.ndarray:
        pass  # Implemented in child class
    
    # Make a batched version of the `transform_signal` function
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
    
    # Make a batched version of the `predict_segmentation` function
    batch_predict_segmentation = vmap(predict_segmentation, in_axes=(None, 0), out_axes=0)

    



    

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

    
    @jax.jit
    def main_loss(self, signal: jnp.ndarray, true_segmentation: jnp.ndarray, lmbda: float) -> float:
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
        transformed_signal = self.transform_signal(signal)
        
        # Calculate the segmentation loss using the transformed signal, penalty parameter, and true segmentation
        segmentation_loss = self.loss(transformed_signal, self.params['beta'], true_segmentation)
        
        # Calculate the regularization term based on the absolute values of the 'omega_weights'
        regularization_term = lmbda * jnp.sum(jnp.abs(self.params['omega_weights']))
        
        # Compute the total main loss by combining the segmentation loss and regularization term
        main_loss_value = segmentation_loss + regularization_term
        
        return main_loss_value

    # Bacthed version of jax's value_and_grad function for main_loss
    self.batched_value_and_grad = jax.jit(vmap(value_and_grad(self.main_loss, argnums=0, allow_int=True), in_axes=(None,0,0,None), out_axes=0 ))


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
        losses, grads = self.batched_value_and_grad(params, signals, true_segmentation, lmbda)
        
        # Compute the final loss as the sum of batched losses
        final_loss = jnp.sum(losses)
        
        # Aggregate gradients for each parameter
        for name, value in grads.items():
            grads[name] = jnp.sum(value, axis=0)
            
        return final_loss, grads

    
    
    
    def train(self, test_signals: jnp.ndarray, strided_true_segmentations: jnp.ndarray,
              num_epochs: int = 100, batch_size: int = 10, test_batch_idx: List[int] = [0],
              start_learning_rate: float = 1e-2, weight_decay: float = 0.001, lmbda: float = 0.0001,
              verbose: bool = True, print_train_accuracy: bool = False) -> Tuple[List[float], optax.OptState, Dict[str, jnp.ndarray]]:
        """
        Trains the scaling network using the given training data.

        Parameters:
        test_signals (jnp.ndarray): Test signals for evaluation.
        strided_true_segmentations (jnp.ndarray): True segmentations corresponding to test signals.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        test_batch_idx (List[int]): List of batch indices to be used for testing.
        start_learning_rate (float): Initial learning rate for optimization.
        weight_decay (float): Weight decay factor for regularization.
        verbose (bool): If True, prints training progress.
        print_train_accuracy (bool): If True, prints training accuracy.
        
        Returns:
        Tuple[List[float], optax.OptState, Dict[str, jnp.ndarray]]:
            - List[float]: List of training losses for each epoch.
            - optax.OptState: The final optimizer state after training.
            - Dict[str, jnp.ndarray]: The final trained parameters of the network.
        """
        # Initialize placeholders.
        train_loss = []
        acc_train = []
        acc_test = []

        # Exponential decay of the learning rate.
        scheduler = optax.exponential_decay(
            init_value=start_learning_rate,
            transition_steps=1000,
            decay_rate=0.99)

        # Combining gradient transforms using `optax.chain`
        gradient_transform = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip the gradients by the global norm.
            optax.scale_by_adam(),          # Use the updates from Adam optimizer.
            optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
            optax.scale(-1.0),              # Scale updates by -1 for gradient descent.
            optax.additive_weight_decay(weight_decay)  # Add weight decay to the updates.
        )

        opt_state = gradient_transform.init(self.params)

        # Loop over the training epochs
        for epoch in range(num_epochs):
            start_time = time.time()
            train_loader = create_data_loader(test_signals, strided_true_segmentations, batch_size, test_batch_idx)
            batch_losses = []
            batch_acc_train = []
            batch_acc_test = []
            for batch_type, (data, target) in train_loader:
                if batch_type == "Train":
                    value, grads = final_loss_and_grad(data, target, lmbda)
                    updates, opt_state = gradient_transform.update(grads, opt_state, params=self.params)
                    self.params = optax.apply_updates(self.params, updates)
                    self.params['omega_weights'] = jnp.maximum(self.params['omega_weights'], 0.0)
                    batch_losses.append(value)
                    if print_train_accuracy:
                        f1_score = bactch_accuracy(self.params, data, target, test_signals[0].shape[0])
                        batch_acc_train.append(f1_score)

                elif batch_type == "Test":
                    if print_train_accuracy:
                        f1_score = bactch_accuracy(self.params, data, target, test_signals[0].shape[0])
                        batch_acc_test.append(f1_score)

            epoch_loss = np.mean(batch_losses)
            train_loss.append(epoch_loss)
            epoch_acc_train = np.mean(batch_acc_train)
            epoch_acc_test = np.mean(batch_acc_test)
            acc_train.append(epoch_acc_train)
            acc_test.append(epoch_acc_test)
            epoch_time = time.time() - start_time
            if verbose:
                print("Epoch {} | T: {:0.2f} | Train A: {:0.3f} | Test A: {:0.3f} | Loss:  {:0.4f} | pen = exp(Beta): {:0.5f}".format(
                    epoch+1, epoch_time, epoch_acc_train, epoch_acc_test, epoch_loss, jnp.exp(self.params['beta'])
                ))

        return train_loss, opt_state, self.params
    






