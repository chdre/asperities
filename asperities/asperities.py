import scipy.ndimage as scpi
import numpy as np


class Asperities:
    def __init__(self, image, allow_split=False):
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
        self.lw, self.num = scpi.measurements.label(image)
        self.allow_split = allow_split
        self.image = image
        self.asperities = None

        roll_label = [[], []]

        for i in range(self.lw[:, 0].shape[0]):
            if self.lw[i, 0] != 0 and self.lw[i, -1] != 0:
                self.lw[self.lw == self.lw[i, -1]] = self.lw[i, 0]
                if self.lw[i, 0] not in roll_label[0]:
                    roll_label[0].append(self.lw[i, 0])

        for i in range(self.lw[0, :].shape[0]):
            if self.lw[0, i] != 0 and self.lw[-1, i] != 0:
                self.lw[self.lw == self.lw[0, i]] = self.lw[-1, i]
                if self.lw[-1, i] not in roll_label[1]:
                    roll_label[1].append(self.lw[-1, i])

        num = np.unique(self.lw[self.lw != 0])
        self.asperities = np.zeros(
            (num.shape[0], self.lw.shape[0], self.lw.shape[1]))

        for i, n in enumerate(num):
            img = self.lw.copy()
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

            self.asperities[i, :, :] = img / np.max(img)
            assert not np.any(self.asperities > 1), 'Overlapping asperities'

    def __call__(self):
        return self.asperities

    def get_image(self):
        return self.asperities.sum(axis=0)

    def shuffle_asperity(self, image, shuffle_factor=1, seed=1):
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

    def shuffle_asperities(self,
                           overlap=False,
                           max_iter=100,
                           shuffle_factor=1,
                           allow_split=True):
        """Shuffle multiple asperities in an image.

        Arguments:
            overlap (bool): Whether to allow overlap when shuffling asperities.
                Defaults to False.
            max_iter (int): Max iterations of while loop. Defaults to 100.
            shuffle_factor (int): Factor of shifting. Defaults to 1.
            allow_split (bool): Whether to allow asperities being split across
                boundaries. Defaults to True.

        Raises:
            ValuError: If there is no possible shuffling without overlap for current
                max_iter.

        Returns:
            image (arr): Image containing shuffled asperities.
        """
        if overlap:
            for i in range(asperities.shape[0]):
                self.asperities[i, :, :] = self.shuffle_asperity(
                    self.asperities[i, :, :])
        else:
            c = 0
            asp_sum = 2  # Dummy to initiate while loop
            while np.any(asp_sum > 1) and c < max_iter:
                for i in range(self.asperities.shape[0]):
                    self.asperities[i, :, :] = self.shuffle_asperity(
                        self.asperities[i, :, :])
                    asp_sum = self.asperities.sum(axis=0)
                c += 1
            if c == max_iter:
                print(np.where(asp_sum > 1), np.any(asp_sum > 1))
                plt.imshow(asp_sum, cmap='gray')
                plt.show()
                plt.imshow(image, cmap='gray')
                plt.show()
                raise ValueError(
                    'Non-overlap not possible when shuffling asperities. Try increasing max_iter.')

        return self.asperities.sum(axis=0)

    def add_asperities(self,
                       overlap: bool = False,
                       overlap_threshold: int = 25,
                       seed: int = 1):
        """ Add asperities along the first axis.

        If overlap is False the array will shuffle whichever asperity that is
        causing the overlapping. The shuffling will continue maximum of
        overlap_threshold steps.

        Arguments:
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
            sum_asperities = self.asperities.sum(axis=0)
        else:
            N = self.asperities.shape[0]
            for i in range(N):
                tmp = self.asperities[:i + 1, :, :].sum(axis=0)
                c = 1
                while np.any(tmp > 1) and c <= overlap_threshold:
                    self.asperities[i, :, :] = self.shuffle_asperity(
                        self.asperities[i, :, :], seed=int(seed * c))
                    tmp = self.asperities[:i + 1, :, :].sum(axis=0)
                    c += 1

                if np.any(tmp > 1):
                    tmp = np.ones(
                        (self.asperities.shape[1], self.asperities.shape[2]))
                    break
            sum_asperities = tmp

        return sum_asperities

    @staticmethod
    def closer(image, step, axes):
        N = image.shape[0]
        assert N == 2, "Method closer only has support for two voids"
        image[0] = np.roll(image[0], step, axes)
        image[1] = np.roll(image[1], -np.asarray(step), axes)

        image = self.add_asperities(image)

        return image

    def move_asperities(self, step, axes, method='closer'):
        """ Moves asperities along given axes. If the goal is to move multiple
        voids independetly of one another, then step and axes can be lists.

        :param image: Image depecting boolean geometry containing one or more
                      voids/asperities
        :type image: ndarray
        :param step: Step length along axes.
        :type step: array_like
        :param axes: List of integer(s) telling which axes to move void/asperity
                     along.
        :type axes: array_like
        :param method: Decide how to move voids
        :type method: str
        :returns: asperities_moved
        :rtype: ndarray
        """
        assert self.num == 2, 'Method only supports 2 voids'

        if method == 'closer':
            self.asperities[0] = np.roll(self.asperities[0], step, axes)
            self.asperities[1] = np.roll(
                self.asperities[1], -np.asarray(step), axes)

        # func = {'closer': closer(image, step, axes)}
        # new_image = func[method]

        new_image = self.add_asperities()

        return new_image

    def change_size(self, image, thickness, type='increase'):
        """Changes the size of voids by increasing it by a set number of pixels.

        :param image: Image depecting boolean geometry containing one or more
                      voids/asperities
        :type image: ndarray
        :param thickness: Thickness given in number of pixels
        :type thickness: int
        :param type: Type of size change, increase or decrease. Defaults to
                     'increase'.
        :type type: str
        :returns new_image:
        :rtype: ndarray
        """
        raise NotImplementedError('Not yet implemented')
        lw, num = scpi.measurements.label(image)

        if num > 1:
            image = get_asperities(image)

        if method == 'increase':
            pass

    def update_asperities(self, array):
        """Changes the asperities to input array.

        :param array: Array representing the new asperities
        :type array: ndarray
        """
        self.asperities = array
