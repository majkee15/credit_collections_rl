def mySearchSorted(array, target):
    """
    Binary search
    Args:
        array:
        target:

    Returns:

    """
    left = 0
    right = len(array)
    while left <= right:
        mid = (left + right) // 2

        if mid == len(array):
            return len(array)

        value_mid = array[mid]

        if array[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left


if __name__ == '__main__':
    import numpy as np
    ar = np.arange(0, 500., 7)
    target = 500

    print(mySearchSorted(ar, target))
    print(np.searchsorted(ar, target, side='right'))
    print(np.digitize(target, ar, right=True))