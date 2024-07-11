import numpy as np
import re, sys, cv2


def create_random_mask(size, num_points=30, seed=0):
    np.random.seed(seed)

    points = np.random.randint(0, 32, size=(num_points, 2))
    hull = cv2.convexHull(points)
    scale_x = size / hull[:,:,0].max()
    scale_y = size / hull[:,:,1].max()
    hull = (hull * np.array([scale_x, scale_y])).astype(int)

    image = np.zeros((size, size), dtype=np.uint8)
    cv2.drawContours(image, [hull], 0, 255, -1)

    _, binary_image = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY)
    new_shape = (size, size) + (1, 3)
    new_arr = np.zeros(new_shape, dtype=binary_image.dtype)
    new_arr[..., :] = binary_image[..., np.newaxis, np.newaxis]

    return new_arr


def create_model(size=32, shape='square'):
    size = int(size)
    model = np.ones([size, size, 1])

    if re.match('square', shape, re.IGNORECASE) is not None:
        print("Create [square] model\n")

    elif re.match('circle', shape, re.IGNORECASE) is not None:
        cc = (size/2 -0.5, size/2 -0.5)
        mx, my = np.meshgrid(np.arange(size), np.arange(size))
        rr = (size//2)**2
        model[ (mx - cc[0])**2 + (my - cc[1])**2 > rr ] = 0
        model[ (mx - cc[0])**2 + (my - cc[1])**2 > rr ] = 0
        print("Create [circle] model\n")

    elif re.match('triangle', shape, re.IGNORECASE) is not None:
        mx, my = np.meshgrid(np.arange(size), np.arange(size))
        model[ mx >  0.5814 * my + 0.5 * size -0.5 ] = 0
        model[ mx < -0.5814 * my + 0.5 * size -0.5 ] = 0
        model[ my > 0.86 * size ] = 0
        print("Create [triangle] model\n")

    else:
        print('Unknown model shape "{}"! Please use one of the folowing:'.format(shape))
        print(' Square  |  Circle  |  Triangle\n')
        sys.exit(0)


    return model


# Custom function to parse a list of floats
def Culist(string):
    try:
        float_values = [float(value) for value in string.split(',')]
        if len(float_values) != 3:
            raise argparse.ArgumentTypeError("List must contain exactly three elements")
        return float_values
    except ValueError:
        raise argparse.ArgumentTypeError("List must contain valid floats")

