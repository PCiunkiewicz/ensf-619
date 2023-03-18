import numpy as np


def centered_circle(image_shape, radius):
    """
    *ADAPTED FROM `https://github.com/rmsouza01/CD-Deep-Cascade-MR-Reconstruction`*
    Description: creates a boolean centered circle image with a pre-defined radius
    :param image_shape: shape of the desired image
    :param radius: radius of the desired circle
    :return: circle image. It is a boolean image
    """
    center_x = image_shape[0] // 2
    center_y = image_shape[1] // 2

    X,Y = np.indices(image_shape)
    circle_image = ((X - center_x)**2 + (Y - center_y)**2) < radius**2 # type: bool

    return circle_image


def gaussian2d(pattern_shape, factor, center=None, cov=None, radius=0, seed=None):
    """
    *ADAPTED FROM `https://github.com/rmsouza01/CD-Deep-Cascade-MR-Reconstruction`*
    Description: creates a 2D gaussian sampling pattern of a 2D image
    :param factor: sampling factor in the desired direction
    :param pattern_shape: shape of the desired sampling pattern.
    :param center: coordinates of the center of the Gaussian distribution
    :param cov: covariance matrix of the distribution
    :param radius: optional centre circle mask with specified radius
    :return: sampling pattern image. It is a boolean image
    """
    random_state = np.random.RandomState(seed=seed)
    N = pattern_shape[0] * pattern_shape[1] # Image length
    factor = int(N * factor)

    if center is None:
        center = np.array([
            1.0 * pattern_shape[0] / 2 - 0.5,
            1.0 * pattern_shape[1] / 2 - 0.5
        ])

    if cov is None:
        cov = np.array([
            [(1.0 * pattern_shape[0] / 4) ** 2, 0],
            [0, (1.0 * pattern_shape[1] / 4) ** 2]
        ])

    m = 1
    samples = np.zeros(0)

    while (samples.shape[0] < factor):
        samples = random_state.multivariate_normal(center, cov, m * factor)
        samples = np.rint(samples).astype(int)
        indexesx = np.logical_and(samples[:, 0] >= 0, samples[:, 0] < pattern_shape[0])
        indexesy = np.logical_and(samples[:, 1] >= 0, samples[:, 1] < pattern_shape[1])
        indexes = np.logical_and(indexesx, indexesy)
        if radius > 0:
            center_x = pattern_shape[0] // 2
            center_y = pattern_shape[1] // 2
            center_mask = ((samples[:, 0] - center_x)**2 + (samples[:, 1]-center_y)**2) < radius**2
            indexes = np.logical_and(indexes, ~center_mask)

        samples = samples[indexes]
        samples = np.unique(samples, axis=0)
        if samples.shape[0] < factor:
            m *= 2

    random_state.shuffle(samples)
    if radius > 0:
        circle = centered_circle(pattern_shape, radius)
        samples = np.vstack((np.argwhere(circle), samples))
    samples = samples[:factor]

    under_pattern = np.zeros(pattern_shape, dtype=bool)
    under_pattern[samples[:, 0], samples[:, 1]] = True
    return under_pattern
