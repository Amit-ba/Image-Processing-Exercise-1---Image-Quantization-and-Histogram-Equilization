import numpy as np
from matplotlib import pyplot as plt
from imageio import imread
from skimage import color as sk
import math

# ======================== [ BASIC FUNCTIONS ] ========================

def read_image(filename, representation):
    """
    Reads image
    :param filename: name of file
    :param representation: 1 for bw, 2 for color
    :return: image, bw or color as wanted, in 0-1 range
    """
    image = imread(filename)
    image = image.astype('float64')
    image = image/255
    if(representation == 1 and len(image.shape) == 3):
        if(image.shape[2] == 3):
            image = sk.rgb2gray(image)
        else:
            # I downloaded some RGBA images for testing so I added this:
            image = sk.rgb2gray(sk.rgba2rgb(image))
    return image

def im_display(filename, representation):
    """
    Displays images
    :param filename: name of file
    :param representation: 1 for bw, 2 for color
    :return: displays image
    """
    image = read_image(filename, representation)
    if(representation == 1): #grey:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
    plt.axis("off")
    plt.show()


def rgb2yiq(imRGB):
    """
    turns rbg image to yiq
    :param imRGB: image in RGB format
    :return: YIQ image
    """
    mat = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])
    result = np.dot(imRGB, mat.T.copy())
    return result

def yiq2rgb(imYIQ):
    """
    turn yiq image to rgb
    :param imYIQ: image in YIQ format
    :return: RGB image
    """
    mat = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])
    mat = np.linalg.inv(mat) # we want the inverse of that matrix
    result = np.dot(imYIQ, mat.T.copy())
    return result



# ======================== [ HISTOGRAM EQUALIZATION ] ========================

def histogram_equalize(im_orig):
    """
    Equalized image
    :param im_orig: image, can be color or bw
    :return: equalized image (=image with more contrast)
    """
    if(len(im_orig.shape) == 3):
        # if this is RGB - we want to work on the Y of YIQ
        yiq_original_image = rgb2yiq(im_orig).copy()
        image = yiq_original_image[:,:,0]
    else:
        image = im_orig.copy()

    list = histogram_equalize_helper(image.copy())

    if(len(im_orig.shape) == 3):
        yiq_equalized_image = list[0]
        final_image = yiq_original_image
        final_image[:,:,0] = yiq_equalized_image #changing only the y channel
        final_image = yiq2rgb(final_image)
        list[0] = final_image

    return list

def histogram_equalize_helper(image):
    """
    Helps equalize an image.
    :param image: a bw image
    :return: an eqalized image
    """
    image = np.around(image*255)
    # Let's calculate some histograms:
    hist, bounds = np.histogram(image, bins=256, range=(0,255))

    cumulative_histogram = np.cumsum(hist)
    pixels_number = image.shape[0] * image.shape[1]
    normalized_cumulative_histogram = cumulative_histogram / pixels_number

    # building lookup table:
    c_m = normalized_cumulative_histogram[np.amin(np.nonzero(normalized_cumulative_histogram))]
    c_255 = normalized_cumulative_histogram[255]
    if(not(int(c_m) == 0 and int(c_255) == 255)):
        lookup_array = np.around(255*((normalized_cumulative_histogram - c_m)/(c_255-c_m))).astype(int)
        image = np.where(image < 0, 0, image)
        image = lookup_array[image.astype(int)]

    equalized_histogram, bounds = np.histogram(image, bins=256, range=(0,255))
    image = image/255
    return [image, hist, equalized_histogram]

# ======================== [ BW QUANTIZATION ] ========================

def quantize(im_orig, n_quant, n_iter):
    """
    Runs the optimal quantization algorithm on photo.
    :param im_orig: an image
    :param n_quant: number of colors in output.
    :param n_iter: max number of iteration for algorithm.
    :return:
    """
    if(len(im_orig.shape) == 3):
        # if this is RGB - we want to work on the Y of YIQ
        yiq_original_image = rgb2yiq(im_orig).copy()
        image = yiq_original_image[:,:,0]
    else:
        image = im_orig.copy()

    list = quantize_helper(image.copy(), n_quant, n_iter)

    if(len(im_orig.shape) == 3):
        yiq_equalized_image = list[0]
        final_image = yiq_original_image
        final_image[:,:,0] = yiq_equalized_image #changing only the y channel
        final_image = yiq2rgb(final_image)
        list[0] = final_image
    return list

def quantize_helper(image, n_quant, n_iter):
    """
    Helps quantize an image.
    :param image: a bw image
    :param n_quant: number of colors in output image
    :param n_iter: number of max iterations
    :return:
    """
    image = np.around(image * 255).astype(int)
    error = []
    hist, bounds = np.histogram(image, bins=256, range=(0,255))
    z_values = guess_initial_z_values(image, n_quant)

    # Looping n-iter times
    for i in range(0, n_iter):
        q_values = compute_q_values(hist, z_values)
        new_z_values = compute_z_values(q_values)
        error.append(compute_error(hist, new_z_values, q_values))
        if(new_z_values == z_values):
            break # if  z hasn't change the limits has converged (or hit a local minimum point)
        else:
            z_values = new_z_values

    image = quantize_remap(image, z_values, q_values)
    image = image/255
    error = np.array(error)
    return [image, error]



def quantize_remap(image, z_values, q_values):
    """
    remaps image according to new colors
    :param image: bw image
    :param z_values: bounds for remap
    :param q_values: new color values for remap
    :return:
    """
    # Looping through n_quant times
    for i in range(1, len(z_values)):
        image[(z_values[i - 1] <= image) & (image <= z_values[i])] = q_values[i - 1]
    return image

def compute_error(hist, z_values, q_values):
    """
    Calaculate error by SSD - sum squared difference.
    :param hist: image histogram
    :param z_values: z values list
    :param q_values: q values list
    :return:
    """
    # the formula we saw in class gives loops over n_quant colors, and in each loop calculates the segment error.
    # this is only so we can connect each original grey value to the new one.
    # but given I can remap all original grey values to new grey values efficently - which i can,
    # if can just do a vectorial operation on all of the grey values at once
    grey_values = np.arange(256)
    quantized_values = quantize_remap(grey_values.copy(), z_values, q_values) #map of every grey value -> q value
    error = ((quantized_values-grey_values)**2)*hist
    return np.sum(error)


def compute_q_values(hist, z_values):
    """
    compute q (=color) values according to algorithm.
    :param hist: histogram of image.
    :param z_values: list of bounds.
    :return:
    """
    q_values = []
    # Looping through n_quant times
    for i in range(len(z_values) - 1):
        z_i = math.floor(z_values[i]) + 1
        if (i == 0): z_i = 0
        z_i_next = math.floor(z_values[i + 1]) + 1 #python doesn't include the last value (which we want to include)
        weights = hist[z_i:z_i_next]
        if(np.sum(weights) ==  0):
            # in this case the denominator may be zero and lead to error.
            # however we were told to crash program only in case of series error
            # this value was chosen to minimize future errors
            q_values.append(0.5*(z_i + z_i_next))
        else:
            q_i = np.average(np.arange(z_i,z_i_next), weights=weights)
            q_values.append(q_i)
    # given there are only so many colors our monitors can view, fractions of q_values hold no meaning
    q_values = [round(q) for q in q_values]
    return q_values
        

def compute_z_values(q_values):
    """
    compute z values according to formula.
    :param q_values: list of q values as defined in algorithm.
    :return: updated z values.
    """
    # the formula for bounds calculations is not mathematically well defined for z_0, z_k.
    # therefore I added them manually.
    lambda_query = lambda i: int(round(0.5 * (q_values[i - 1] + q_values[i])))
    z_values = [lambda_query(i) for i in range(1, len(q_values))] #looping through n_quant times
    z_values = [0] + z_values + [255]
    return z_values



def guess_initial_z_values(image, n_quant):
    """
    guesses initial z values.
    :param image: bw image
    :param n_quant: number of bins (= number of colors  in final image)
    :return: list of z values
    """
    # there are qute a few ways to write this fuctions, e.g. with np.quantile.
    # However this was the one that seems to be best - in some cases, especially ones with
    # low contrast, this function gives me less error than the quantization algorithm itself.
    x = image.flatten()
    # returns evenly spread (nbin + 1) numbers in [0, img_pixel_num] segment:
    x_coordinates = np.linspace(0, len(x), n_quant + 1)
    xp = np.arange(len(x)) # all numbers 0...len(x)
    fp = np.sort(x)
    z_values = list(np.interp(x_coordinates, xp, fp))
    if(float(z_values[0]) != float(0)):
        z_values[0] = float(0)
    if(float(z_values[-1]) != float(255)):
        z_values[-1] = float(255)
    return z_values





