"""
Mask R-CNN
Train on the toy Chromosome dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla (modified by Andrew Cox)

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 chromosome.py train --dataset=/path/to/chromosome/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 chromosome.py train --dataset=/path/to/chromosome/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 chromosome.py train --dataset=/path/to/chromosome/dataset --weights=imagenet

    # Apply color splash to an image
    python3 chromosome.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 chromosome.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import time
import datetime
import numpy as np
import skimage.draw
import PIL.Image as Image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import COCO Tools 
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class ChromosomeConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "chromosome"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + chromosome

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class ChromosomeDataset(utils.Dataset):

    def load_chromosome(self, dataset_dir, subset, return_coco=False):
        """Load a subset of the Chromosome dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        chroms = COCO("{}/chromosome.json".format(dataset_dir))
        print(chroms)
        image_dir = "{}/".format(dataset_dir)

        # Load class IDs
        class_ids = sorted(chroms.getCatIds())
        print('class IDs {}'.format(class_ids))
        # Add classes. We have only one class to add.
        # self.add_class("chromosome", 1, "chromosome")

        # Grab images
        image_ids = []
        for id in class_ids:
            image_ids.extend(list(chroms.getImgIds(catIds=[id])))
        print(image_ids)
        # Remove duplicates
        image_ids = list(set(image_ids))
        
        # Add classes
        for i in class_ids: 
            self.add_class("chroms", i, chroms.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "chroms", image_id=i,
                path=os.path.join(image_dir, chroms.imgs[i]['file_name']),
                width=chroms.imgs[i]["width"],
                height=chroms.imgs[i]["height"],
                annotations=chroms.loadAnns(chroms.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        
        if return_coco:
            return chroms

        # # Load annotations
        # # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # # { 'filename': '28503151_5b5b7ec140_b.jpg',
        # #   'regions': {
        # #       '0': {
        # #           'region_attributes': {},
        # #           'shape_attributes': {
        # #               'all_points_x': [...],
        # #               'all_points_y': [...],
        # #               'name': 'polygon'}},
        # #       ... more regions ...
        # #   },
        # #   'size': 100202
        # # }
        # # We mostly care about the x and y coordinates of each region
        # # Note: In VIA 2.0, regions was changed from a dict to a list.
        # annotations = json.load(open(os.path.join(dataset_dir, "chromosome.json")))
        # annotations = list(annotations.values())  # don't need the dict keys

        # # The VIA tool saves images in the JSON even if they don't have any
        # # annotations. Skip unannotated images.
        # annotations = [a for a in annotations if a['regions']]

        # # Add images
        # for a in annotations:
        #     # Get the x, y coordinaets of points of the polygons that make up
        #     # the outline of each object instance. These are stores in the
        #     # shape_attributes (see json format above)
        #     # The if condition is needed to support VIA versions 1.x and 2.x.
        #     if type(a['regions']) is dict:
        #         polygons = [r['shape_attributes'] for r in a['regions'].values()]
        #     else:
        #         polygons = [r['shape_attributes'] for r in a['regions']] 

        #     # load_mask() needs the image size to convert polygons to masks.
        #     # Unfortunately, VIA doesn't include it in JSON, so we must read
        #     # the image. This is only managable since the dataset is tiny.
        #     image_path = os.path.join(dataset_dir, a['filename'])
        #     image = skimage.io.imread(image_path)
        #     height, width = image.shape[:2]

        #     self.add_image(
        #         "chromosome",
        #         image_id=a['filename'],  # use file name as a unique image id
        #         path=image_path,
        #         width=width, height=height,
        #         polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Chromosome image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "chroms":
            return super(ChromosomeDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "chroms.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "chromosome":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        # completely_arbitrary = ann['image_id']
        # completely_arbitrary2 = ann['id']
        # print('segmentation {}, height {}, width {}, image_id {}, id {})'.format(segm, height, width, completely_arbitrary, completely_arbitrary2))
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


###################################################
# Evaluation
###################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "chroms"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


###################################################
# Training
###################################################
def train(model):
    # """Train the model."""

    # import imgaug # for augmentation

    # # Training dataset.
    # dataset_train = ChromosomeDataset()
    # dataset_train.load_chromosome(args.dataset, "train")
    # dataset_train.prepare()

    # # Validation dataset
    # dataset_val = ChromosomeDataset()
    # dataset_val.load_chromosome(args.dataset, "val")
    # dataset_val.prepare()

    # # *** This training schedule is an example. Update to your needs ***
    # # Since we're using a very small dataset, and starting from
    # # COCO trained weights, we don't need to train too long. Also,
    # # no need to train all layers, just the heads should do it.
    # print("Training network heads")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=50,
    #             augmentation = imgaug.augmenters.Sometimes(0.5, [
    #                 imgaug.augmenters.flip.Fliplr(0.5)
    #                 # imgaug.augmenters.flip.Flipud(0.5),
    #                 #imgaug.augmenters.GaussianBlur(sigma=(0.0, 2.0))
    #             ]),
    #             layers="3+")

    """Train the model."""
    # Training dataset.
    dataset_train = ChromosomeDataset()
    dataset_train.load_chromosome(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ChromosomeDataset()
    dataset_val.load_chromosome(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    # print("Training all network layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE ,
                epochs=20,
                layers='heads')

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image, plugin = 'pil')
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect chromosome.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/chromosome/dataset/",
                        help='Directory of the Chromosome dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--limit', required=False, 
                        default=50, 
                        metavar="<image count>", 
                        help="Images to use for evaluation (default=50)")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    elif args.command == 'evaluate':
        # Configurations
        class InferenceConfig(ChromosomeConfig):
            # Set batch size to 1 since we'll be running inference on 
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
        config.display()
        
        # Create model
        model = modellib.MaskRCNN(mode='inference', config=config, 
                                  model_dir=args.logs)

        # Select weights file to load
        if args.weights.lower() == "coco":
            model_path = COCO_MODEL_PATH
        elif args.weights.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()
        elif args.weights.lower() == "imagenet":
            # Start from ImageNet trained weights
            model_path = model.get_imagenet_weights()
        else:
            model_path = args.model

        # Load weights
        print("Loading weights", model_path)

        # Validation dataset
        dataset_val = ChromosomeDataset()
        chrom = dataset_val.load_chromosome(args.dataset, "val", return_coco=True)
        dataset_val.prepare()
        print("Running COCO evaluation on {} chromosome images.".format(args.limit))
        evaluate_coco(model, dataset_val, chrom, eval_type='segm', limit=int(args.limit))

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ChromosomeConfig()
    else:
        class InferenceConfig(ChromosomeConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
