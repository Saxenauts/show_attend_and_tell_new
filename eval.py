"""
USAGE: python eval.py /path/to/images
"""

import os, os.path
import sys
from PIL import Image
import numpy as np
from core.vggnet import Vgg19
from scipy import ndimage

from core.new_solver import CaptioningSolver
from core.new_model import CaptionGenerator
from core.utils import *

model_path = 'model/lstm_hard/model-40'


#Take the image, and preprocess it to get VGG features
#Extract features of the images
#A function to resize the images
def resize_image(image):
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = width - left
        top = 0
        bottom = height
    else:
        top = (height - width) / 2
        bottom = height - top
        left = 0
        right = width
    image = image.crop((left, top, right, bottom))
    image = image.resize([224, 224], Image.ANTIALIAS)
    return image

def extract_features(images_resized):
    vggnet = Vgg19(vgg_model_path)
    vggnet.build()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        n_images = len(image_resized)
        all_feats = np.ndarray([n_images, 196, 512], dtype=np.float32)

        all_feats = sess.run(vggnet.features, feed_dict = {
                                vggnet.images: image_batch})

        return all_feats

def main(images_path):
    #image_filenames
    image_filenames = [name for name in os.listdir(images_path) if os.path.isfile(name)]
    #number of images
    n = len(image_filenames)
    images_resized = np.zeros([224, 224, 3])
    for i, image_path in enumerate(image_filenames):
        image = ndimage.imread(image_path, mode = 'RGB').astype(np.float32)
        image = resize_image(image)
        images_resized[i] = image

    #Extracted VGG features of all the images
    all_feats = extract_features(images_resized)

    #Now Generate captions
    data = load_coco_data(data_path='./data', split='val', if_train=True)
    word_to_idx = data['word_to_idx']
    model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=512,
                             dim_hidden=1024, n_time_step=16, prev2out=True,
                             ctx2out=True, alpha_c=1.0, selector=True, dropout=True)
    solver = CaptioningSolver(model, data, data, n_epochs=10, batch_size=100, update_rule='adam',
                              learning_rate=0.001, print_every=1000, save_every=5, image_path='./image/',
                              pretrained_model=None, model_path='model/lstm_hard/', test_model='model/lstm_hard/model-40',
                              print_bleu=True, log_path='log/')

    alphas, betas, sampled_captions = solver.model.build_sampler(
        max_len=20)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, solver.test_model)
        feed_dict = {solver.model.features: all_feats}

        alps, bts, sam_cap = sess.run(
            [alphas, betas, sampled_captions], feed_dict)
        decoded = decode_captions(sam_cap, solver.model.idx_to_word)


        all_sam_cap = np.ndarray((all_feats.shape[0], 20))
        for i in range(n):
            feed_dict = {solver.model.features: all_feats}
            all_sam_cap[i] = sess.run(sampled_captions, feed_dict)
            all_decoded = decode_captions(all_sam_cap, solver.model.idx_to_word)
            print "Generated caption for image", image_filenames[i]
            print all_decoded[i]

if __name__ = '__main__':
    main(sys.argv[1])
