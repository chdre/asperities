from scipy import ndimage
import numpy as np


class Asperities:
    def __init__(self, image, allow_split=True):
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
        # We also consider asperities that are diagonally touching to be a
        # a single void. The line below ensures this is the case.
        s = ndimage.generate_binary_structure(2, 2)

        self.lw, self.num = ndimage.label(image, structure=s)
        self.allow_split = allow_split
        self.image = image
        self.asperities = None

        roll_label = [[], []]

        # Correct for PBC, give correct name to labeled image. From POV of x = 0
        # we look at x = N_x to check if there is also an asperity there, if
        # there is, we know that these are connected and we gather them both
        # under the same label
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
                _, num_tmp = ndimage.label(img)

                if num_tmp == 0:
                    raise ValueError('No labels in the image')

                if num_tmp > 1:  # If the asperity is split into two we must roll the array
                    if n in roll_label[0]:
                        c = 0
                        while ndimage.label(img)[1] != 1 and c < img.shape[0] // 2:
                            img = np.roll(img, shift=1, axis=1)
                            c += 1
                            if n in roll_label[1]:
                                c2 = 0
                                while ndimage.label(img)[1] != 1 and c2 < img.shape[1] // 2:
                                    img = np.roll(img, shift=1, axis=0)
                                    c2 += 1
                                roll_label[1].remove(n)
                    if n in roll_label[1]:
                        c = 0
                        while ndimage.label(img)[1] != 1 and c < img.shape[1] // 2:
                            img = np.roll(img, shift=1, axis=0)
                            c += 1

            self.asperities[i, :, :] = img / np.max(img)
            assert not np.any(self.asperities >
                              1), 'Overlapping asperities'

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

    def move_asperities(self, step, axes, method='closer', **kwargs):
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

        new_image = self.add_asperities(kwargs)

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
        lw, num = ndimage.label(image)

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

    @staticmethod
    def center_of_mass(asperities, max_search):
        """Calculates the center of mass (COM) for each asperity.

        :param asperities: Array containing asperities along axis 0.
        :type asperities. ndarray
        :param max_search: Maximum number of searches when rolling asperites.
        :type max_search: int
        :returns com: Array containing COM coordinates
        :rtype com: ndarray
        """

        labeled, n_labels = ndimage.label(asperities)

        com_original = ndimage.center_of_mass(asperities, asperities, 1)

        roll = 0
        c = 0
        asperities_copy = asperities  # Create copy to use for rolling
        while n_labels > 1 and c < max_search:
            # Roll array if the asperity is split, until it is no longer split,
            # if possible. Could be that there is a void spanning entire space
            roll += 1
            asperities = np.roll(asperities_copy, (roll, roll), axis=(0, 1))
            labeled, n_labels = ndimage.label(asperities)

            c += 1

        if c >= max_search and n_labels > 1:
            # If we did not find a new one we assume the asperity stretches
            # across all boundaries
            com = com_original
        else:
            # If the loop above broke naturally we use the last created rolled
            # asperities object and unroll the COM
            com = ndimage.center_of_mass(asperities, asperities, 1)
            com = np.array(com) - roll
            if com[0] > asperities.shape[0]:
                com[0] -= asperities.shape[0]
            if com[0] < -asperities.shape[0]:
                com[0] += asperities.shape[0]
            if com[1] > asperities.shape[1]:
                com[1] -= asperities.shape[1]
            if com[1] < -asperities.shape[1]:
                com[1] += asperities.shape[1]

        return com

    def get_center_of_masses(self, pbc=True, tile_factor=(1, 1), max_search=100):
        """Fetch the center of mass for each void in the asperity.

        :param pbc: Whether or not to account for periodic boundary conditions.
                    Defaults to True.
        :type pbc: bool
        :param tile_factor: If tiling the asperities we tile by these factors.
                            I.e. the factors to multiply the image along
                            respective axes. Defaults to (1, 1), i.e. no tiling.
        :param max_search: Maximum number of searches when rolling asperites.
                           Defaults to 100.
        :type max_search: int
        :returns com: Center of masses for asperity
        :rtype com: ndarray
        """
        asp = self.asperities.copy()  # To avoid overwriting
        if pbc:
            asp = np.tile(asp, tile_factor)

        com = np.array([self.center_of_mass(a, max_search) for a in asp])

        return com

    def radial_distribution(self, dr, pbc=True, tile_factor=(1, 1),
                            prob_density=True, max_search=100):
        """Finds the radial distribution of voids by the center of mass for each
        separate void. If pbc (periodic boundary conditions) is true the image
        will be repeated to account for periodicity. If we do not account for
        periodicity we will calculate the distance between a single void for the
        case where a voids is split across one or more boundaries.


        :param dr: Radius for each shell in the radial distribution. Defaults
                   to None, which means the dr is automatically calculated by
                        dr = maximum radius / number of labels.
                   Should be fixed if one is working with multiple inputs.
        :type dr: float
        :param pbc: Whether or not to account for periodic boundary conditions.
                    Defaults to True.
        :type pbc: bool
        :param tile_factor: If tiling the asperities we tile by these factors.
                            I.e. the factors to multiply the image along
                            respective axes. Defaults to (1, 1), i.e. no tiling.
        :type tile_factor: array_like
        :param prob_density: Tells whether to return the probability density of
                             finding an asperity in a spherical shell.
                             Probability density is found by
                                P = count / (sum(count) * dr)
                             where dr is the width of each spherical shell.
                             Should be set to False if one is working with
                             multiple inputs.
        :type prob_density: bool
        :param max_search: Maximum number of searches when rolling asperites
                           when finding center of masses. Defaults to 100.
        :type max_search: int
        :returns shell_counter: Number of asperities in each shell, or
                                probability density.
        :rtype shell_counter: np.ndarray
        :returns shells: Interval of shells.
        :rtype shells: np.ndarray
        """
        asp = self.asperities.copy()  # To avoid overwriting

        if pbc:
            asp = np.tile(asp, tile_factor)

        # Max radius is the the one which spans from origo to opposite vertice
        # i.e. sqrt( (0, 0)**2 + (200, 100) ** 2 )
        max_radius = np.linalg.norm(asp.shape[1:])

        shells = np.concatenate(
            (np.arange(0, max_radius, dr), np.array([max_radius])), axis=0)

        # Find which asperity sits in what shell
        com = np.array([self.center_of_mass(a, max_search) for a in asp])

        # COM POV: counter for the other COMs in own shells
        shells_counter = np.zeros((shells.shape[0] - 1))

        # Checking each COM (i.e. each asperity) seperately
        for i, c in enumerate(com):
            # Find the distance to all asperities from POV of a specific asperity
            dist = np.linalg.norm(
                np.delete(com, i, axis=0) - c[np.newaxis, :], axis=1)

            # Find how many asperities are in each shell from POV of specific asperity
            shells_counter += np.array([np.logical_and(dist >= shells[j], dist <
                                       shells[j + 1]).sum() for j in range(shells.shape[0] - 1)])

        if prob_density:
            # Normalizing and dividing by the width of the spherical shells.
            # This makes sure that the AUC (area under curve) is 1, i.e. the
            # function returns the probability of finding an asperity within a shell
            shells_counter /= (shells_counter.sum() * dr)

        return shells_counter, shells
