import numpy as np

def find_closest_index(sorted_list: list[float], target_value: float) -> int:
    """
    Find the index of the element in a sorted array that is closest to a target value.

    Parameters:
    sorted_array (list[float]): A sorted list of values.
    target_value (float): The value for which the closest element's index is to be found.

    Returns:
    int: The index of the closest element to the target value in the sorted_array.
    """

    # Convert the sorted_array to a NumPy array for efficient computations
    sorted_array = np.asarray(sorted_list)
    
    # Use binary search to find the index where the target_value should be inserted in sorted_array
    idx = np.searchsorted(sorted_array, target_value)
    
    # If target_value should be inserted at the beginning of sorted_array
    if idx == 0:
        return 0
    # If target_value should be inserted at the end of sorted_array
    elif idx == len(sorted_array):
        return len(sorted_array) - 1
    else:
        # Calculate the absolute differences between target_value and the elements on both sides of idx
        left_diff = np.abs(sorted_array[idx - 1] - target_value)
        right_diff = np.abs(sorted_array[idx] - target_value)
        
        # Compare the absolute differences and choose the index with the smallest difference
        if left_diff <= right_diff:
            return idx - 1
        else:
            return idx

