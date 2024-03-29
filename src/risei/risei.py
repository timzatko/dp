import time
import cv2

import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage.restoration import inpaint_biharmonic

from multiprocessing import Pool

from PIL import Image


def generate_mask(params):
    grid = params['grid']
    options = params['options']
    i = params['i']
    image_data = params['image_data']
    shift_x, shift_y, shift_z = params['random_shift']

    # mask has a soft corners
    mask = __get_mask(options, grid, shift_x, shift_y, shift_z)
    # binary mask does not have a soft corners and is used for an in_painting
    binary_mask = __get_binary_mask(options, grid, shift_x, shift_y, shift_z)

    in_paint_mask = None
    if options['b1'] > 0:
        in_paint_mask = __get_in_paint_mask(options, image_data, mask, binary_mask)

    # print(mask.shape)
    new_image = __merge(options, image_data, mask, in_paint_mask)

    cache = None
    # when the debug mode is enabled
    # save the generated images to cache
    if options['debug']:
        cache = {
            'grid': grid,
            'binary_mask': binary_mask,
            'in_paint_mask': in_paint_mask,
        }

    return i, new_image, mask, cache


def __merge(options, original_image, mask, in_paint_mask):
    # original_image = (original_image - original_image.min())

    # blend original image with in_paint mask if exists
    new_image = original_image

    if in_paint_mask is not None:
        if options['b1'] < 1:
            new_image = (1 - options['b1']) * original_image + options['b1'] * in_paint_mask
        else:
            new_image = in_paint_mask

    # blend original image with in_paint mask with mask
    if options['b2'] > 0:
        if options['b2_value'] != 0:
            value = 1
            if options['b2_value'] == 'mean':
                value = np.mean(original_image)
            elif options['b2_value'] == 'median':
                value = np.median(original_image)
            new_image = (mask * options['b2']) * new_image + ((1 - mask) * value * options['b2'])
        else:
            new_image = (mask * options['b2']) * new_image

    return new_image


def __get_in_paint_mask(options, image_data, mask, binary_mask):
    if options['in_paint'] == '3d':
        return __get_in_paint_mask_3d(options, image_data, mask, binary_mask)
    return __get_in_paint_mask_2d(options, image_data, mask, binary_mask)


def __get_in_paint_mask_3d(options, image_data, mask, binary_mask):
    start = time.time()

    output = np.zeros(image_data.shape)
    inverted_binary_mask = 1 - binary_mask.astype(np.uint8)
    in_painted = inpaint_biharmonic(image_data, inverted_binary_mask, multichannel=False);

    if options['in_paint_blending']:
        # in_paint with gradual blending of edges (soft edges)
        output = image_data * mask + in_painted * (1 - mask)

    end = time.time()
    print(f"in: {end - start}")

    return output


def __get_in_paint_mask_2d(options, image_data, mask, binary_mask):
    start = time.time()
    in_painted = np.zeros(image_data.shape)
    inverted_binary_mask = (1 - binary_mask).astype(np.uint8)
    
    if options['in_paint_2d_to_3d'] is False:
        for z in range(0, image_data.shape[0]):
            in_painted_z = cv2.inpaint(
                image_data[z],
                inverted_binary_mask[z],
                options['in_paint_radius'],
                options['in_paint_algorithm']
            )
            
            if options['in_paint_blending']:
                # in_paint with gradual blending of edges (soft edges)
                in_painted_z = image_data[z] * mask[z] + in_painted_z * (1 - mask[z])

            in_painted[z] = in_painted_z
    else:
        for i in range(0, image_data.shape[0]):
            in_painted_i = cv2.inpaint(
                image_data[i, :, :],
                inverted_binary_mask[i, :, :],
                options['in_paint_radius'],
                options['in_paint_algorithm']
            )
            in_painted[i, :, :] += in_painted_i

        for i in range(0, image_data.shape[1]):
            in_painted_i = cv2.inpaint(
                image_data[:, i, :],
                inverted_binary_mask[:, i, :],
                options['in_paint_radius'],
                options['in_paint_algorithm']
            )
            in_painted[:, i, :] += in_painted_i

        for i in range(0, image_data.shape[2]):
            in_painted_i = cv2.inpaint(
                image_data[:, :, i],
                inverted_binary_mask[:, :, i],
                options['in_paint_radius'],
                options['in_paint_algorithm']
            )
            in_painted[:, :, i] += in_painted_i
                
        in_painted /= 3
        
        if options['in_paint_blending']:
            # in_paint with gradual blending of edges (soft edges)
            in_painted = image_data * mask + in_painted * (1 - mask)

    end = time.time()

    # print(f"in: {end - start}")

    return in_painted


def __get_mask(options, grid, shift_x, shift_y, shift_z):
    return resize(
        grid,
        options['mask_size'],
        order=1,
        mode='reflect',
        anti_aliasing=False)[
        shift_x:shift_x + options['input_size'][0],
        shift_y:shift_y + options['input_size'][1],
        shift_z:shift_z + options['input_size'][2]
    ]


def __get_binary_mask(options, grid, shift_x, shift_y, shift_z):
    new_grid = np.zeros(options['mask_size'])
    input_size = options['input_size']
    start = time.time()

    for a in range(0, grid.shape[0]):
        for b in range(0, grid.shape[1]):
            for c in range(0, grid.shape[2]):
                x = a * options['cell_size'][0]
                y = b * options['cell_size'][1]
                z = c * options['cell_size'][2]

                new_grid[x:x + options['cell_size'][0], y:y + options['cell_size'][1],
                z:z + options['cell_size'][2]] = int(grid[a][b][c])

    end = time.time()
    # print(f"gbm: {end - start}")

    return new_grid[shift_x:input_size[0] + shift_x, shift_y:input_size[1] + shift_y, shift_z:input_size[2] + shift_z]


def get_image(image, dim, i):
    if dim == 0:
        return image[i, :, :]
    elif dim == 1:
        return image[:, i, :]
    elif dim == 2:
        return image[:, :, i]
    else:
        raise Exception('dimension should be >= 0 and <= 2')


class RISEI:
    def __init__(self, input_size, **kwargs):
        self.options = {
            'input_size': np.array(input_size, dtype=np.uint),
            's': kwargs.get('s', 8),  # size of the "grid" - binary mask
            'p1': kwargs.get('p1', 0.5),  # probability of cell being white - transparent
            'b1': kwargs.get('b1', 0.8),  # in_paint mask blend
            'b2': kwargs.get('b2', 0.5),  # black mask blend
            'in_paint': kwargs.get('in_paint', '2d'),  # 3d, 2d
            'in_paint_radius': kwargs.get('in_paint_radius', 20),  # in_painting radius
            'in_paint_algorithm': kwargs.get('in_paint_algorithm', cv2.INPAINT_NS),  # cv2.INPAINT_TELEA, cv2.INPAINT_NS
            'in_paint_blending': kwargs.get('in_paint_blending', True),
            'in_paint_2d_to_3d': kwargs.get('in_paint_2d_to_3d', False),
            # if the in_paint is gradually blended into the image
            'debug': kwargs.get('debug', False),
            'b2_value': kwargs.get('b2_value', 0), # 0, 1, (min, max, mean, meadian) - implement
            'mask_size': None,
            'cell_size': None,
            'over_image_size': None,
            'processes': kwargs.get('processes', 4),
        }

        self.cache = None

        self.__get_grid_size()

    def generate_masks(self, n, image, log=True, seed=None):
        # if we are setting a new seed, save the original seed
        if seed is not None:
            st0 = np.random.get_state()
            np.random.seed(seed)

        self.__initialize_cache(n, image)

        grids = self.__get_empty_grids(n)
        images_data = self.__get_empty_images_data(n)
        images_mask = self.__get_empty_images_data(n)
        random_shift = self.__get_random_shifts(n)

        # print(image.shape)
        
        params = [
            {
                'i': i,
                'options': self.options,
                'grid': grids[i],
                'random_shift': random_shift[i],
                'image_data': image
            } for i in range(0, n)]

        # use process pool only when more than a one process is used
        # in google colaboratory it causes memory leaks
        if self.options['processes'] > 1:
            process_pool = Pool(processes=self.options['processes'])

            with process_pool as p:
                with tqdm(desc='Generating masks', total=n, disable=not log) as bar:
                    for i, new_image, mask, cache in p.imap_unordered(generate_mask, params):
                        images_data[i, :, :, :] = new_image
                        images_mask[i, :, :, :] = mask

                        if cache is not None:
                            self.__save_to_cache(i, new_image, mask, cache)

                        bar.update()

            process_pool.close()
        else:
            with tqdm(desc='Generating masks', total=n, disable=not log) as bar:
                for i, new_image, mask, cache in map(generate_mask, params):
                    images_data[i, :, :, :] = new_image
                    images_mask[i, :, :, :] = mask

                    if cache is not None:
                        self.__save_to_cache(i, new_image, mask, cache)

                    bar.update()

        # set back original random seed
        if seed is not None:
            np.random.set_state(st0)

        return images_data, images_mask

    def show_from_last_run(self, i, z, figsize=(12, 8), ncols=3, nrows=2, dim=0):
        original_image = self.show_image_from_last_run(i, z, dim)
        mask = self.show_mask_from_last_run(i, z, dim)
        binary_mask = self.show_binary_mask_from_last_run(i, z, dim)
        in_paint = self.show_in_paint_from_last_run(i, z, dim) if self.options[
                                                                      'b1'] > 0 else self.show_image_from_last_run(i,
                                                                                                                   z,
                                                                                                                   dim)
        result = self.show_result_from_last_run(i, z, dim)

        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
        ax = axes.ravel()

        ax[0].set_title('Original')
        ax[0].imshow(original_image)

        ax[1].set_title('Binary mask')
        ax[1].imshow(binary_mask)

        ax[2].set_title('Mask')
        ax[2].imshow(mask)

        ax[3].set_title('in_paint')
        ax[3].imshow(in_paint)

        ax[4].set_title('Result')
        ax[4].imshow(result)

        for a in ax:
            a.axis('off')

        fig.tight_layout()
        plt.show()

    def show_mask_from_last_run(self, i, z, dim):
        mask = self.__get_from_cache('masks', i)

        new_image = get_image(mask, dim, z)
        image = 255 * np.ones((*new_image.shape, 3), dtype=np.uint8)
        new_image = image * new_image.reshape((*new_image.shape, 1))

        return Image.fromarray(new_image.astype(np.uint8), 'RGB')

    def show_binary_mask_from_last_run(self, i, z, dim):
        binary_mask = self.__get_from_cache('binary_masks', i)

        new_image = get_image(binary_mask, dim, z)
        image = 255 * np.ones((*new_image.shape, 3), dtype=np.uint8)
        new_image = image * new_image.reshape((*new_image.shape, 1))

        return Image.fromarray(new_image.astype(np.uint8), 'RGB')

    def show_image_from_last_run(self, i, z, dim):
        if self.cache is None:
            raise Exception('Cache is not defined! Initialize algorithm with debug=True')

        return get_image(self.cache['image'], dim, z)

    def show_in_paint_from_last_run(self, i, z, dim):
        return get_image(self.__get_from_cache('in_paint_masks', i), dim, z)

    def show_result_from_last_run(self, i, z, dim):
        return get_image(self.__get_from_cache('images_data', i), dim, z)

    def __get_from_cache(self, key, i):
        if self.cache is None:
            raise Exception('Cache is not defined! Initialize algorithm with debug=True')
        if len(self.cache[key]) <= i:
            raise Exception(f'Index {i} does not exist!')
        return self.cache[key][i]

    def __save_to_cache(self, i, image_data, mask, cache):
        self.cache['images_data'][i] = image_data
        self.cache['masks'][i] = mask
        self.cache['grids'][i] = cache['grid']
        self.cache['binary_masks'][i] = cache['binary_mask']
        self.cache['in_paint_masks'][i] = cache['in_paint_mask']

    def __initialize_cache(self, N, image):
        if self.options['debug']:
            self.cache = {
                'image': image,
                'images_data': np.empty((N, *self.options['input_size'])),
                'grids': np.empty((N, self.options['s'], self.options['s'], self.options['s'])),
                'masks': np.empty((N, *self.options['input_size'])),
                'binary_masks': np.empty((N, *self.options['input_size'])),
                'in_paint_masks': np.empty((N, *self.options['input_size'])),
            }
        else:
            self.cache = None

    def __get_empty_grids(self, N):
        grids = np.random.rand(N, self.options['s'], self.options['s'], self.options['s']) < self.options['p1']
        return grids.astype('float32')

    def __get_random_shifts(self, N):
        return np.array([np.array(np.random.rand(3) * self.options['over_image_size'], dtype=np.uint) for _ in range(N)])
    
    def __get_empty_images_data(self, N):
        return np.empty((N, *self.options['input_size']))

    def __get_grid_size(self):
        # the size of one pixel (rectangle)
        cell_size = np.ceil(np.array(self.options['input_size']) / self.options['s'])

        # the additional size for each rectangle
        # since we do a crop from the mask, we
        # need to make a "bigger" mask
        # we calculate how much we need to increase the cell
        # to increase the current size by one additional cell
        over_cell_size = np.ceil(
            (((self.options['s'] + 1) * cell_size) - (self.options['s'] * cell_size)) / self.options['s'])

        # new cell size
        new_cell_size = cell_size + over_cell_size

        # mask size (mask is larger than the image)
        mask_size = (self.options['s'] * new_cell_size).astype(np.uint32)

        # difference in size between original image and mask
        over_image_size = mask_size - self.options['input_size']

        self.options['mask_size'] = mask_size
        self.options['cell_size'] = new_cell_size.astype(np.uint32)
        self.options['over_image_size'] = over_image_size
