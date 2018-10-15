import numpy as np
import math
import sys

sigma = 1.4

'''
The neighborhood is only vertically, horizontally, but not diagonally
Don't worry, The last direction can be Extend.
'''





def boundary_term_exponential(intensities):
    """
    Implementation of a exponential boundary term computation over an array.
    """
    # apply exp-(x**2/sigma**2)
    intensities = np.power(intensities, 2)
    intensities /= math.pow(sigma, 2)
    intensities *= -1
    intensities = np.exp(intensities)
    intensities[intensities <= 0] = sys.float_info.min
    return intensities

def intensity_difference(neighbour_one, neighbour_two):
    """
    Takes two voxel arrays constituting neighbours and computes the absolute
    intensity differences.
    """
    return np.absolute(neighbour_one - neighbour_two)

def skeleton_base(graph, image, boundary_term=boundary_term_exponential,
                  neighbourhood_function=intensity_difference, spacing=False):
    """
    Base of the skeleton for voxel based boundary term calculation.

    This function holds the low level procedures shared by nearly all boundary terms.

    @param graph An initialized graph.GCGraph object
    @type graph.GCGraph
    @param image The image containing the voxel intensity values
    @type image numpy.ndarray
    @param boundary_term A function to compute the boundary term over an array of
                           absolute intensity differences
    @type boundary_term function
    @param neighbourhood_function A function that takes two arrays of neighbouring pixels
                                  and computes an intensity term from them that is
                                  returned as a single array of the same shape
    @type neighbourhood_function function
    @param spacing A sequence containing the slice spacing used for weighting the
                   computed neighbourhood weight value for different dimensions. If
                   False, no distance based weighting of the graph edges is performed.
    @param spacing sequence | False
    """
    image = image.astype(np.float)

    # iterate over the image dimensions and for each create the appropriate edges and compute the associated weights
    for dim in range(image.ndim):
        # construct slice-objects for the current dimension
        slices_exclude_last = [slice(None)] * image.ndim
        slices_exclude_last[dim] = slice(-1)
        slices_exclude_first = [slice(None)] * image.ndim
        slices_exclude_first[dim] = slice(1, None)
        # compute difference between all layers in the current dimensions direction
        neighbourhood_intensity_term = neighbourhood_function(image[slices_exclude_last], image[slices_exclude_first])

        # record
        print('Neibohood:')
        print(neighbourhood_intensity_term)

        # apply boundary term
        neighbourhood_intensity_term = boundary_term(neighbourhood_intensity_term)

        # record
        print('Neibohood intensity term:')
        print(neighbourhood_intensity_term)

        # compute key offset for relative key difference
        offset_key = [1 if i == dim else 0 for i in range(image.ndim)]
        offset = flatten_index(offset_key, image.shape)
        # generate index offset function for index dependent offset
        idx_offset_divider = (image.shape[dim] - 1) * offset
        idx_offset = lambda x: int(x / idx_offset_divider) * offset

        # weight the computed distanced in dimension dim by the corresponding slice spacing provided
        if spacing: neighbourhood_intensity_term /= spacing[dim]

        for key, value in enumerate(neighbourhood_intensity_term.ravel()):
            # apply index dependent offset
            key += idx_offset(key)
            # add edges and set the weight
            print('Add nweight: From %d, to %d, forward weight %f, backward weight %f'%
                  (key, key+offset, value, value))
            # graph.set_nweight(key, key + offset, value, value) # !


def flatten_index(pos, shape):
    """
    Takes a three dimensional index (x,y,z) and computes the index required to access the
    same element in the flattened version of the array.
    """
    res = 0
    acc = 1
    for pi, si in zip(reversed(pos), reversed(shape)):
        res += pi * acc
        acc *= si
    return res

def test():
    img = np.random.randint(20, size=(4,5))
    print(img)
    skeleton_base(None, img)

if __name__ == '__main__':
    test()