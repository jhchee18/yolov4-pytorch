
Diamond-Back Moths - v1 DBM Junjie Set
==============================

This dataset was exported via roboflow.com on September 18, 2022 at 8:29 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

It includes 1562 images.
DBM are annotated in YOLO v4 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 416x416 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip

The following transformations were applied to the bounding boxes of each image:
* Random brigthness adjustment of between -32 and 0 percent


