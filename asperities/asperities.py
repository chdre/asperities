import scipy.ndimage as scpi
import numpy as np


def get_asperities(image, allow_split=False):
    """Find the asperities of a given image.

    Arguments:
        image (arr): Array containing image(s).
        allow_split (bool): If to correct for split asperities.

    Raises:
        ValueError: If the image does not contain any asperities, and therefore
            has no labels.

    Returns:
        asperities (arr): Array containing all the asperities.
    """
    lw, num = scpi.measurements.label(image)

    roll_label = [[], []]

    for i in range(lw[:, 0].shape[0]):
        if lw[i, 0] != 0 and lw[i, -1] != 0:
            lw[lw == lw[i, -1]] = lw[i, 0]
            if lw[i, 0] not in roll_label[0]:
                roll_label[0].append(lw[i, 0])

    for i in range(lw[0, :].shape[0]):
        if lw[0, i] != 0 and lw[-1, i] != 0:
            lw[lw == lw[0, i]] = lw[-1, i]
            if lw[-1, i] not in roll_label[1]:
                roll_label[1].append(lw[-1, i])

    num = np.unique(lw[lw != 0])
    asperities = np.zeros((num.shape[0], lw.shape[0], lw.shape[1]))

    for i, n in enumerate(num):
        img = lw.copy()
        img[img != n] = 0
        if not allow_split:
            _, num_tmp = scpi.measurements.label(img)

            if num_tmp == 0:
                raise ValueError('No labels in the image')

            if num_tmp > 1:  # If the asperity is split into two we must roll the array
                if n in roll_label[0]:
                    c = 0
                    while scpi.measurements.label(img)[1] != 1 and c < img.shape[0] // 2:
                        img = np.roll(img, shift=1, axis=1)
                        c += 1
                        if n in roll_label[1]:
                            c2 = 0
                            while scpi.measurements.label(img)[1] != 1 and c2 < img.shape[1] // 2:
                                img = np.roll(img, shift=1, axis=0)
                                c2 += 1
                            roll_label[1].remove(n)
                if n in roll_label[1]:
                    c = 0
                    while scpi.measurements.label(img)[1] != 1 and c < img.shape[1] // 2:
                        img = np.roll(img, shift=1, axis=0)
                        c += 1

        asperities[i, :, :] = img / np.max(img)
        if np.any(asperities > 1):
            print('fake')

    return asperities


def shuffle_asperity(image, shuffle_factor=1, seed=1):
    """Shuffles labeled blob by rolling the array.

    Arguments:
        image (arr): Labeled image
        shuffle_factor (int): Factor of shifting. Defaults to 1.
        seed (int): Numpy seed for random choice.

    Returns:
        image (arr): Shuffled image.
    """
    np.random.seed(seed)
    Nx, Ny = image.shape * shuffle_factor
    Nx, Ny = int(Nx), int(Ny)
    x, y = np.random.choice(
        range(-Nx, Nx), size=1)[0], np.random.choice(range(-Ny, Ny), size=1)[0]
    image = np.roll(image, shift=(x, y), axis=(0, 1))

    return image


def shuffle_asperities(image,
                       overlap=False,
                       max_iter=100,
                       shuffle_factor=1):
    """Shuffle multiple asperities in an image.

    Arguments:
        image (arr): Array containing image values.
        overlap (bool): Whether to allow overlap when shuffling asperities.
            Defaults to False.
        max_iter (int): Max iterations of while loop. Defaults to 100.
        shuffle_factor (int): Factor of shifting. Defaults to 1.

    Raises:
        ValuError: If there is no possible shuffling without overlap for current
            max_iter.

    Returns:
        image (arr): Image containing shuffled asperities.
    """
    asperities = get_asperities(image, allow_split=True)
    if overlap:
        for i in range(asperities.shape[0]):
            asperities[i, :, :] = shuffle_asperity(asperities[i, :, :])
    else:
        c = 0
        asp_sum = 2  # Dummy to initiate while loop
        while np.any(asp_sum > 1) and c < max_iter:
            for i in range(asperities.shape[0]):
                asperities[i, :, :] = shuffle_asperity(asperities[i, :, :])
                asp_sum = asperities.sum(axis=0)
            c += 1
        if c == max_iter:
            print(np.where(asp_sum > 1), np.any(asp_sum > 1))
            plt.imshow(asp_sum, cmap='gray')
            plt.show()
            plt.imshow(image, cmap='gray')
            plt.show()
            raise ValueError(
                'Non-overlap not possible when shuffling asperities. Try increasing max_iter.')

    return asperities.sum(axis=0)


def add_asperities(asperities: np.ndarray,
                   overlap: bool = False,
                   overlap_threshold: int = 25,
                   seed: int = 1):
    """ Add asperities along the first axis.

    If overlap is False the array will shuffle whichever asperity that is
    causing the overlapping. The shuffling will continue maximum of
    overlap_threshold steps.

    Arguments:
        asperities: Images along the second and third axis. First axis contains the number of images.
        overlap: Whether to allow asperities to overlap when adding.
        overlap_threshold: Maximum tries for overlap shuffling.
        seed: Seed for shuffling asperity.

    Raises:
        ValueError: If shuffling does not resolve issue of overlapping.

    Returns:
        sum_asperities: Summed array.

    TO DO:
        Add support for asperities being array_like, not just ndarray
    """
    if overlap:
        sum_asperities = asperities.sum(axis=0)
    else:
        N = asperities.shape[0]
        for i in range(N):
            tmp = asperities[:i + 1, :, :].sum(axis=0)
            c = 1
            while np.any(tmp > 1) and c <= overlap_threshold:
                asperities[i, :, :] = shuffle_asperity(
                    asperities[i, :, :], seed=int(seed * c))
                tmp = asperities[:i + 1, :, :].sum(axis=0)
                c += 1

            if np.any(tmp > 1):
                tmp = np.ones((asperities.shape[1], asperities.shape[2]))
                break
        sum_asperities = tmp

    return sum_asperities
