# import required modules

import re         # for regular expression matching
import random
import numpy as np
import os
import scipy.misc
import shutil     # high-level operations for files and collections of files
import zipfile    # module: provides tools to create, read, write, append zip files.
import time
import tensorflow as tf
from glob import glob   # find all the pathnames matching a specified pattern according to rules by unix.
from urllib.request import urlretrieve # high-level interface for fetching data across the WWW
from tqdm import tqdm  # progress bar printer


class DLProgress(tqdm):
    """
    Report download progress to the terminal
    """
    last_block = 0
    
    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        help.
        """
        self.total_size = total_size
        self.update((block_num-self.last_block) * block_size) # update progress
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist.
    :param data_dir: Directory to download the model to.
    """
    vgg_filename = "vgg.zip"
    vgg_path = os.path.join(data_dir, "vgg")
    vgg_files = [
                 os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
                 os.path.join(vgg_path, 'variables/variables.index'),
                 os.path.join(vgg_path, 'saved_model.pb')
                ]

    
    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.isfile(vgg_file)]

    if missing_vgg_files:
        # if there are missing files, clean up the directory if exists
        if os.path.isdir(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)


        print("Download the pretrained model...")
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                        "https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip",
                        os.path.join(vgg_path, filename),
                        pbar.hook)

        print("Extracting model...")
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref = zipfile.extract_all(data_dir)
        zip_ref.close()


    # remove zip file
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function_backup(data_folder, image_shape):
    """Generate function to create batches of training data."""

    image_paths = glob(os.path.join(data_folder, "image_2", "*.png"))

    # label is a dictionary

    # os.path.basename(path): return the basename. Eg. /foo/bar --> bar
    # image_path: re.sub(r'_(lane|road)_', '_', os.path.basename(path))
    # gt_path: ..................................

    label_paths = {
                   re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path for path in glob(os.path.join(data_folder, 'gt_image_2', '_road_*.png'))}

    background_color = np.array([255, 0, 0])   # background color is red (GT)
    
    # shuffle training data
    random.shuffle(image_paths)
    
    # loop through batches, yield each batch.
    for batch_i in range(0, len(image_paths), batch_size):
        images = []
        gt_images = []
        for image_file in image_paths[batch_i: batch_i+batch_size]:
            gt_image_file = label_paths[os.path.basename(image_file)]
            # resize
            image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
            
            gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
            

            # one-hot label for each pixel
            # test whether all array elements along a given axis evaluate to True.
            # compare along the color channel.
            gt_bg = np.all(gt_image == background_color, axis=2)
            gt_bg = gt_bg.reshape(*gt_bg.shape, 1)

            # ground truth, channel=0 denotes background, channel=1 denotes foreground.
            gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

            images.append(image)
            gt_images.append(gt_image)

        yield np.array(images), np.array(gt_images)


def gen_batch_function(data_folder, image_shape):
    """Generate function to create batches of training data."""


    # label is a dictionary

    # os.path.basename(path): return the basename. Eg. /foo/bar --> bar
    # image_path: re.sub(r'_(lane|road)_', '_', os.path.basename(path))
    # gt_path: ..................................
    
    def get_batches_fn(batch_size): 

        image_paths = glob(os.path.join(data_folder, "image_2", "*.png"))
        label_paths = {
                   re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}

        background_color = np.array([255, 0, 0])   # background color is red (GT)
    
    # shuffle training data
        random.shuffle(image_paths)
    
    # loop through batches, yield each batch.
       
        # print("label_path", label_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i: batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]
            # resize
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
            
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
            

            # one-hot label for each pixel
            # test whether all array elements along a given axis evaluate to True.
            # compare along the color channel.
                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)

            # ground truth, channel=0 denotes background, channel=1 denotes foreground.
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)


    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):    
    for image_file in glob(os.path.join(data_folder, "image_2", "*.png")):
        
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        
        # inference
        im_softmax = sess.run([tf.nn.softmax(logits)], feed_dict={keep_prob: 1.,
                                                                  image_pl: [image]})
        # channel = 1 denotes foreground
        im_softmax = im_softmax[0][:, 1].reshape(*image_shape) 

        # true foreground
        segmentation = (im_softmax > 0.5).reshape(*image_shape, 1)
        mask = np.multiply(segmentation, np.array([[0, 255, 0, 127]]))
        # mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode='RGBA')
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)
        


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    # RUN nn on testing images
    print("Training Finished. Saving testing to: {}".format(output_dir))
    image_outputs = gen_test_output(
                                    sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    # print("len(image_outputs) ={}".format(image_outputs))
    for name, image in image_outputs:
        # print("counter = {}".format(counter))
        scipy.misc.imsave(os.path.join(output_dir, name), image)



if __name__ == '__main__':
    data_dir = './data'
    data_folder = os.path.join(data_dir, 'data_load/testing')
    print(data_folder)
    for image_file in glob(os.path.join(data_folder, "image_2", "*.png")):
        print(image_file)
        print(os.path.isfile(image_file))

