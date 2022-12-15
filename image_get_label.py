#!/usr/bin/env python
# coding: utf-8
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

#import pyrealsense2 as rs
import numpy as np
from PIL import Image
import cv2
import time
import random
import os
from strawberry_stem_detect import get_label_from_imgdataset

class Imageprocess:
    #只能保存单个的
    def __init__(self,save_path):
        # self.strawberry_imgpath = strawberry_imgpath
        # self.strawberry_labelpath = strawberry_labelpath
        # self.stem_imgpath = stem_imgpath
        # self.stem_labelpath = stem_labelpath
        self.save_path = save_path


    def img_process(self,img_input_color, file_name):
        img_input_color = img_input_color[:,:,(2,1,0)]
        self.strawberryimg = Image.fromarray(cv2.cvtColor(img_input_color, cv2.COLOR_BGR2RGB))
        result = get_label_from_imgdataset(self.strawberryimg)
        if result is None:
            self.strawberrybox = None
            self.bbox_save(file_name)
            img_input_color = img_input_color[:, :, (2, 1, 0)]
            return img_input_color

        imageout, self.strawberrybox, self.classes = result
        
        self.imgname = 'IMG'+str(time.time()) + '.' + str(random.randint(1000, 9999))

        #print('strawberry bounding box:')
        #print(self.strawberrybox)
            
        self.strawberrybox = self.modify_label(self.strawberryimg, self.strawberrybox)
        
        #self.image_save(file_name)
        self.bbox_save(file_name)
        imageout = np.array(imageout)
        imageout = imageout[:,:,(2,1,0)]
        return imageout

    def modify_label(self,img,bbox):
        height, width = img.size
        bbox_out_all = []
        print(bbox.size)
        for i in range((int) (bbox.size/4)):
            top, left, bottom, right = bbox[i]
            centerw = (left+right)/width/2
            centerh = (top+bottom)/height/2
            bboxw = (right-left)/width
            bboxh = (bottom-top)/height
            #0 is class
            bbox_out = [self.classes[i], centerh, centerw, bboxh, bboxw]
            #bbox_out = [self.classes[i], top, left, bottom, right]
            bbox_out=[' '.join(str(x) for x in bbox_out)]
            bbox_out_all.append(bbox_out)
        return bbox_out_all


    def bbox_save(self,file_name):
        #input: image,bbox
        #output: strawberry bbox, stem bbox. Image name, bbox, classes.
        label_txt = ""
        if (not self.strawberrybox is None):
            for i in self.strawberrybox:
                label_txt += i[0] + "\n"
        #np.savetxt(self.save_path + '/test_label/'+ file_name[0:-4] + '.txt', label_txt, fmt='%s')
        fp = open(self.save_path + '/../test_label_15122022/'+ file_name[0:-4] + '.txt', 'w')
        fp.write(label_txt)
        fp.close()
        
        #if not self.stembox is None:
        #    np.savetxt(self.save_path + '/stem/label/' + file_name[0:-4] + '.txt', self.stembox, fmt='%s')


if __name__ == '__main__':
    # Create image class
    save_path='./dataset/test_15122022'
    improcess = Imageprocess(save_path)
    directory = save_path



    for file_name in os.listdir(directory):
        f = os.path.join(directory, file_name)
        #print(f)
        color_img = Image.open(f)
        
        np_color_image = np.asanyarray(color_img)
        color_image = improcess.img_process(np_color_image, file_name)
        
    print("done!")


    
    '''
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        print('The Resolution is 960*540')
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        print('The Resolution is 640*480')

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

'''






'''
    try:
        while True:

            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            ###############################################################
            # Customized code
            # Get img and box. Publish messages
            color_image = improcess.img_process(color_image)
            # color_image = cv2.line(color_image, (320, 0), (320, 480), (0, 0, 0), thickness=1)
            # color_image = cv2.line(color_image, (0, 240), (640, 240), (0, 0, 0), thickness=1)
            ################################################################

            # If depth and color resolutions are different, resize color image to match depth image for display
            # if depth_colormap_dim != color_colormap_dim:
            #     resized_depth_image = cv2.resize(depth_colormap, dsize=(color_colormap_dim[1], color_colormap_dim[0]),
            #                                      interpolation=cv2.INTER_AREA)
            #     images = np.hstack((color_image, resized_depth_image))
            # else:
            #     images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            cv2.waitKey(10)
    except KeyboardInterrupt:
        pass

    finally:
        # Stop streaming
        cv2.destroyAllWindows()
        pipeline.stop()
        
        '''

