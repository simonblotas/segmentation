import numpy as np
from typing import Tuple
from jax import jit, lax
from jax import numpy as jnp
from jax.tree_util import Partial

def find_closest_index(sorted_list: list[float], target_value: float) -> int:
    """
    Find the index of the element in a sorted array that is closest to a 
    target value.

    Parameters:
    sorted_array (list[float]): A sorted list of values.
    target_value (float): The value for which the closest element's index is 
    to be found.

    Returns:
    int: The index of the closest element to the target value in the 
    sorted_array.
    """

    # Convert the sorted_array to a NumPy array for efficient computations
    sorted_array = np.asarray(sorted_list)
    
    # Use binary search to find the index where the target_value should be 
    # inserted in sorted_array
    idx = np.searchsorted(sorted_array, target_value)
    
    # If target_value should be inserted at the beginning of sorted_array
    if idx == 0:
        return 0
    # If target_value should be inserted at the end of sorted_array
    elif idx == len(sorted_array):
        return len(sorted_array) - 1
    else:
        # Calculate the absolute differences between target_value and the 
        # elements on both sides of idx
        left_diff = np.abs(sorted_array[idx - 1] - target_value)
        right_diff = np.abs(sorted_array[idx] - target_value)
        
        # Compare the absolute differences and choose the index with the 
        # smallest difference
        if left_diff <= right_diff:
            return idx - 1
        else:
            return idx

@jit
def prepend_row_of_zeros(array2d: jnp.ndarray) -> jnp.ndarray:
    """Return the 2D array with a row of 0s at the beginning."""
    return jnp.pad(array=array2d, pad_width=((1, 0), (0, 0)))



def find_last_top_k_changes(
    carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    accu_and_end: Tuple[jnp.ndarray, jnp.int32],
    n_largest: jnp.int32,
    accumulation_arr: jnp.ndarray,
    penalty: jnp.float32,
) -> jnp.ndarray:
    """
    path_arr: at position (i, k), contains the index of the last change-point of
        one of the k-th best segmentations of signal[:i]
    rank_arr: at position (i, k), contains the rank of the last change-point (of
        signal[:i]) path_arr[i, k]  TODO: not sure about this one
    soc_arr: at position (i, k), the sum of costs of the segmentation whose last
        change is path_arr[i, k]
    accumulation_arr: at position t, contains the vector of cumulative sums of
        signal until index t
    duration_vec:
    penalty (float): the penalty level of the change-point detection procedure
    """
    (accumulation_vec, end) = accu_and_end
    (
        path_arr,
        rank_arr,
        soc_arr,
        duration_vec,
    ) = carry

    cost_vec = (
        jnp.square(accumulation_vec - accumulation_arr).sum(axis=1) / duration_vec
    )
    cost_vec = (cost_vec - penalty)[:, None]

    soc_vals, soc_inds = lax.top_k((soc_arr + cost_vec).flatten(), k=n_largest)
    last_bkps, ranks = jnp.unravel_index(indices=soc_inds, shape=soc_arr.shape)

    # update carry
    carry = (
        path_arr.at[end].set(last_bkps),
        rank_arr.at[end].set(ranks),
        soc_arr.at[end].set(soc_vals),
        jnp.roll(duration_vec, shift=1),
    )
    return carry, None


@Partial(jit, static_argnames=["n_largest"])
def get_path_arr(
    signal: jnp.ndarray, penalty: jnp.float32, n_largest: jnp.int32
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return a path array for the top-k segmentations of signal.

    Return (path array, rank array, top sums of costs)
    """
    signal = signal - signal.mean(axis=0)
    # init
    n_samples = signal.shape[0]
    path_arr_shape = (n_samples + 1, n_largest)
    path_arr = jnp.empty(shape=path_arr_shape, dtype=int).at[0].set(-1)
    rank_arr = jnp.empty(shape=path_arr_shape, dtype=int).at[0].set(-1)
    soc_arr = (
        jnp.full(shape=path_arr_shape, fill_value=-jnp.inf, dtype=jnp.float32)
        .at[0, 0]
        .set(0.0)
    )
    accumulation_arr = prepend_row_of_zeros(jnp.cumsum(signal, axis=0))
    duration_vec = jnp.roll(jnp.arange(n_samples + 1).at[0].set(1)[::-1], shift=2)
    carry_init = (
        path_arr,
        rank_arr,
        soc_arr,
        duration_vec,
    )

    find_last_top_k_changes_specialized = Partial(
        find_last_top_k_changes,
        n_largest=n_largest,
        accumulation_arr=accumulation_arr,
        penalty=penalty,
    )

    # actual computation
    (path_arr, rank_arr, soc_arr, _), _ = lax.scan(
        f=find_last_top_k_changes_specialized,
        init=carry_init,
        xs=(accumulation_arr[1:], jnp.arange(1, n_samples + 1)),
    )

    return path_arr, rank_arr, soc_arr[n_samples]





def activate_last_bkp(
    carry: Tuple[jnp.ndarray, jnp.int32, jnp.int32],
    path_arr: jnp.ndarray,
    rank_arr: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.int32, jnp.int32]:
    """Set to 1 the activation vector at the last breakpoint location"""
    # unpack
    (activations, bkp, rank_col) = carry
    # find last breakpoint
    bkp, rank_col = path_arr[bkp, rank_col], rank_arr[bkp, rank_col]
    # activate the last breakpoint
    activations = activations.at[bkp].set(1)
    # carry over
    carry = (activations, bkp, rank_col)
    return carry


def continue_on_path(carry: Tuple[jnp.ndarray, jnp.int32, jnp.int32]) -> jnp.bool_:
    """If current breakpoint is positive, we continue."""
    _, bkp, _ = carry
    return bkp > 0


def get_activations_single_rank(
    rank: jnp.int32,
    path_arr: jnp.ndarray,
    rank_arr: jnp.ndarray,
) -> jnp.ndarray:
    """Return the activation vector associated with the k-th best segmentation"""
    n_samples = path_arr.shape[0] - 1
    carry_init = (
        jnp.zeros(n_samples, dtype=int),  # activation vector
        n_samples,  # convention: last change-point is always the number of samples
        rank,
    )
    activate_last_bkp_specialized = Partial(
        activate_last_bkp, path_arr=path_arr, rank_arr=rank_arr
    )
    activations_single_rank, *_ = lax.while_loop(
        cond_fun=continue_on_path,
        body_fun=activate_last_bkp_specialized,
        init_val=carry_init,
    )
    activations_single_rank = activations_single_rank.at[0].set(0)
    return activations_single_rank  # shape (n_samples,)


@jit
def get_activations(
    path_arr: jnp.ndarray,
    rank_arr: jnp.ndarray,
) -> jnp.ndarray:
    """Return the activation vectors associated with the top-k segmentations

    An activation vector is 1 at a change-point and 0 otherwise.
    """
    get_activations_single_rank_specialized = Partial(
        get_activations_single_rank, path_arr=path_arr, rank_arr=rank_arr
    )

    n_ranks = rank_arr.shape[1]
    activations = lax.map(get_activations_single_rank_specialized, jnp.arange(n_ranks))
    return activations  # shape (n_ranks, n_samples)


@jit
def get_labels(path_arr: jnp.ndarray, rank_arr: jnp.ndarray) -> jnp.ndarray:
    """Return the label vectors associated with the top-k segmentations

    A label vector has constant value on each segment of the top-k segmentations.
    """
    activations = get_activations(path_arr=path_arr, rank_arr=rank_arr)
    return jnp.cumsum(activations, axis=1)


@jit
def projection_unit_simplex(sorted_vec: jnp.ndarray, regularization=1.0) -> jnp.ndarray:
    """Projection onto the unit simplex

    Taken from https://github.com/google/jaxopt/blob/1c16c4322ca5caaeb8b3731618fc1288adb2e4d4/jaxopt/_src/projection.py
    but we assume that the input vector is already sorted in descending order.
    """
    sorted_vec = sorted_vec / regularization
    n_features = sorted_vec.shape[0]
    cumsum_u = jnp.cumsum(sorted_vec)
    ind = jnp.arange(n_features) + 1
    cond = 1.0 / ind + (sorted_vec - cumsum_u / ind) > 0
    idx = jnp.count_nonzero(cond)
    return jax.nn.relu(1.0 / idx + (sorted_vec - cumsum_u[idx - 1] / idx))


@jit
def projection_pwc(signal: jnp.ndarray, segment_ids: jnp.ndarray) -> jnp.ndarray:
    """Project a signal on the space of piecewise constant (pwc) signals.

    The segmentation of the projection is given by segment_ids which is an array
    of increasing integers, constant on each segment.

    signal: shape (n_samples, n_dims)
    segment_ids: shape (n_samples,)
    output: shape (n_samples, n_dims)
    """
    # NOTE: could be faster by setting `num_segments` to a low value
    seg_sums = segment_sum(
        data=signal,
        segment_ids=segment_ids,
        indices_are_sorted=True,
        unique_indices=False,
        num_segments=signal.shape[0],
    )
    one_vec = jnp.ones_like(segment_ids)
    seg_durations = segment_sum(
        data=one_vec,
        segment_ids=segment_ids,
        indices_are_sorted=True,
        unique_indices=False,
        num_segments=signal.shape[0],
    )[:, None]
    return (seg_sums / seg_durations)[segment_ids]


@jit
def get_optimal_projection(
    signal: jnp.ndarray, penalty: float
) -> Tuple[jnp.ndarray, int]:
    path_arr, rank_arr, soc_vec = get_path_arr(
        signal=lax.stop_gradient(signal), penalty=lax.stop_gradient(penalty), n_largest=1
    )
    # projections on the top segmentation of the original signal
    # shape (n_samples, n_dims)

    segment_ids_pred = get_labels(path_arr=path_arr, rank_arr=rank_arr)
    n_segments_pred = segment_ids_pred[0,-1] + 1
    projected_pred = projection_pwc(signal, segment_ids_pred[0])
    return projected_pred, n_segments_pred,  segment_ids_pred.astype(jnp.float32)



def get_segment_ids(segmentation: list[int], signal_length: int) -> jnp.ndarray:
    '''
    Converts a segmentation list [t_0, ..., t_m] into a binary activation array
    of size signal_length. The activation array marks the segments as active (1)
    based on the provided segmentation points.

    Parameters:
    segmentation (list[int]): A list of segmentation points [t_0, ..., t_m].
    signal_length (int): The length of the signal.

    Returns:
    jnp.ndarray: An array representing segment activations with size signal_length.
    '''

    # Initialize an array of zeros representing no activation
    activation = jnp.zeros((signal_length))
    
    # Set the values at segmentation points to 1, indicating segment activation
    activation = activation.at[segmentation].set(1)
    
    # Calculate the cumulative sum of the activation array along the axis
    # This creates an array where each position represents the segment ID
    segment_ids = jnp.cumsum(activation, axis=0, dtype=int)
    
    return segment_ids

@jax.jit
def segmentation_to_projection(signal: jnp.ndarray, segment_ids: jnp.ndarray) -> jnp.ndarray:
    '''
    Calculates the projection of a signal onto its segments, based on the associated
    segmentation.

    Parameters:
    signal (jnp.ndarray): The signal to be projected.
    segment_ids (jnp.ndarray): An array representing segment IDs for each time step.

    Returns:
    jnp.ndarray: The projection of the signal onto its segments.
    '''

    # Calculate the projection using the projection_pwc function
    projection = projection_pwc(signal, segment_ids)
    
    return projection
