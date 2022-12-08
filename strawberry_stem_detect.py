import cv2
import os
import colorsys
import numpy as np
import tkinter.filedialog
from PIL import Image, ImageDraw
#from yolostem import Yolostem
#from yolostrawberry import Yolostrawberry
from yolo_get_label import YOLO
#from yolostrawberry_real import Yolostrawberry
#图像原点在左上角

image_dir = "img/valid"
crop_save_dir = 'img/imgcrop'

#yolo_stage1 = Yolostrawberry()
#yolo_stage2 = Yolostem()

yolo = YOLO()


def pil(img):
    #转灰度图，转binary
    #不同图片threshold不一样
    r,img,b = img.split()
    img = img.convert('L')
    threshold  = 110
    table  =  []
    for  i  in  range( 256 ):
         if  i  <  threshold:
            table.append(0)
         else :
            table.append(1)
    img = img.point(table,'1')
    return img

def get_xy(img):
    x = []
    y = []
    matrix = np.array(img)
    #图片旋转90度，防止梯度过高，matrix.shape的宽高本来就是反的
    w,h = matrix.shape
    for i in range(w):
        for j in range(h):
            #图片左上角为原点，向下为x，向右为y
            if matrix[i,j] == False:
                x.append(i)
                y.append(h-1-j)
    return x, y

def linear_regression(x,y):
    x = np.array(x)
    y = np.array(y)
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x**2)
    sumxy = sum(x*y)
    A = np.mat([[N,sumx],[sumx,sumx2]])
    b = np.array([sumy,sumxy])
    return np.linalg.solve(A,b)

def plotline(img,w,b):
    width,height = img.size
    pointy = int(b+w*height)
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    b = int(b)
    if w>0:
        draw.line((b,height,pointy,0),'red')
    else:
        draw.line((pointy,height, b, 0), 'red')
    return img

def hough(img):
    #输入图像为pil格式
    imgcv = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
    #取绿色通道
    (b,g,r) = cv2.split(imgcv)
    #腐蚀
    kernel = np.asarray([[0, 1, 0], [0, 1, 0], [0, 1, 0]], np.uint8)
    erosion= cv2.erode(g,kernel, iterations=5)
    #canny边缘提取
    edges = cv2.Canny(erosion,100,300,apertureSize=5)

    maxLineGap = 10
    minLineLength = 50
    lines = cv2.HoughLines(edges,1,np.pi/180,10,minLineLength,maxLineGap)
    #判断一个ndarray数组是否为空
    if lines is None:
        return

    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        #乘1000为了以x0y0延长
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(imgcv, (x1, y1), (x2, y2), (0, 255, 255), 3)
        w = -b/a

    # (b, g, r) = cv2.split(imgcv)
    # img = cv2.merge((r,g,b))
    # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # pylab.show()

    img = Image.fromarray(cv2.cvtColor(imgcv,cv2.COLOR_BGR2RGB))
    #img.show()

    return img,w,rho,theta

def find_direction(img,type):
    #如果拓展linear输出过少，会报错
    try:
        if type == 'linear':
            imgt = img.crop((img.size[0] * 0.1, 0, img.size[0] * 0.9, img.size[1] * 0.7))
            imgt = pil(imgt)
            xt, yt = get_xy(imgt)
            b, w = linear_regression(xt, yt)
            img = plotline(img, w, b)
            return img,w
        elif type == 'hough':
            img,w,rho,theta = hough(img)
            return img, w,rho,theta
    except:
        w = 0
        theta = 0
        rho= img.size[0]/2
        return img,w,rho,theta

def crop_bbox(image,boxes):
    top, left, bottom, right = boxes

    # top = top - 30
    # left = left - 20
    # bottom = bottom + 20
    # right = right + 20

    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
    right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

    stem_image = image.crop((left, top, right, bottom))
    return stem_image

def draw_rectangle(img,finalbox,bias,tan,color_index):
    top, left, bottom, right = finalbox
    left = int(left+bias[0])
    right = int(right+bias[0])
    top = int(top+bias[1])
    bottom = int(bottom +bias[1])
    #生成颜色
    # h,s,v = [color_index/10, 1., 1.]
    # colors = list(colorsys.hsv_to_rgb(h,s,v))
    # colors =(int(colors[0] * 255), int(colors[1] * 255), int(colors[2] * 255))
    colors = (205,0,0)
    #生成背景和方框
    background = Image.new('RGB', ((right-left),(bottom-top)), colors)
    squaremask = Image.new('RGBA', ((right-left),(bottom-top)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(squaremask)
    draw.rectangle((0,0,(right-left),(bottom-top)), outline=(255, 255, 255), width=3)
    background = background.rotate(int(tan * 45),expand=True)
    leftpaste = left + squaremask.size[0]/2 -background.size[0]/2
    toppaste = top + squaremask.size[1]/2 -background.size[1]/2
    squaremask = squaremask.rotate(int(tan * 45), expand=True)
    img.paste(background, (int(leftpaste), int(toppaste)), mask=squaremask)
    return img

"""
def get_pick_img(image_in):
    #input image type:np
    #output the stem area in strawberry picture
    image_pre = image_in.copy()
    image = image_in.copy()
    draw = ImageDraw.Draw(image)
    boxes = yolo_stage1.detect_boxes(image_pre)
    # the boxes are set by np.array([None]). Not means the boxes is None.
    if boxes is None:
        return 
    x, y = boxes.shape
    box_out = []
    for i in range(x):
        strawberry_image = crop_bbox(image_pre, boxes[i])
        rtg = (boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2])
        draw.rectangle(rtg, outline=(30, 144, 255), width=4)
        # Second stage: find the stem
        stem_boxes = yolo_stage2.detect_boxes(strawberry_image)
        # the boxes are set by np.array([None]). Not means the boxes is None.
        if stem_boxes is None:
            continue
        #print('stem bounding box:')
        #print(stem_boxes)
        xt, yt = stem_boxes.shape
        for j in range(xt):
            # get the stem direction
            stem_image = crop_bbox(strawberry_image, stem_boxes[j])
            stem_img_line, w,rho,theta = find_direction(stem_image, 'hough')
            im_paste_left = int(np.maximum(boxes[i][1], 0) + np.maximum(stem_boxes[j][1], 0))
            im_paste_top = int(np.maximum(boxes[i][0], 0) + np.maximum(stem_boxes[j][0], 0))
            image.paste(stem_img_line, (im_paste_left, im_paste_top))
            bias = (np.maximum(boxes[i][1], 0), np.maximum(boxes[i][0], 0))
            image = draw_rectangle(image, stem_boxes[j], bias, w, j)
            #draw = ImageDraw.Draw(image)
            #rtg = (boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2])
            #draw.rectangle(rtg, outline=(30, 144, 255), width=4)
            top, left, bottom, right = stem_boxes[j]
            left = left + bias[0]
            right = right + bias[0]
            top = top + bias[1]
            bottom = bottom + bias[1]
            stem_box_out=np.asarray([top, left, bottom, right])
            box_out.append([boxes[i],stem_box_out])
    if box_out:
        return image,box_out
    else:
        print("find strawberry but not find stem")
        return image,box_out

def get_img_label(image_in):
    #input image type:np
    #output the stem area in strawberry picture
    image = image_in.copy()
    boxes = yolo_stage1.detect_boxes(image)
    # the boxes are set by np.array([None]). Not means the boxes is None.
    if boxes is None:
        return
    #print('strawberry bounding box:')
    #print(boxes)
    x, y = boxes.shape
    for i in range(x):
        strawberry_image = crop_bbox(image, boxes[i])
        # Second stage: find the stem
        stem_boxes = yolo_stage2.detect_boxes(strawberry_image)
        # the boxes are set by np.array([None]). Not means the boxes is None.
        if stem_boxes is None:
            continue
        #print('stem bounding box:')
        #print(stem_boxes)
        xt, yt = stem_boxes.shape
        for j in range(xt):
            # get the stem direction
            stem_image = crop_bbox(strawberry_image, stem_boxes[j])
            stem_img_line, w,rho,theta = find_direction(stem_image, 'hough')
            im_paste_left = int(np.maximum(boxes[i][1], 0) + np.maximum(stem_boxes[j][1], 0))
            im_paste_top = int(np.maximum(boxes[i][0], 0) + np.maximum(stem_boxes[j][0], 0))
            image.paste(stem_img_line, (im_paste_left, im_paste_top))
            bias = (np.maximum(boxes[i][1], 0), np.maximum(boxes[i][0], 0))
            image = draw_rectangle(image, stem_boxes[j], bias, w, j)
            draw = ImageDraw.Draw(image)
            rtg = (boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2])
            draw.rectangle(rtg, outline=(30, 144, 255), width=4)
            # 只返回第一个草莓的梗
            top, left, bottom, right = stem_boxes[j]
            left = int(left + bias[0])
            right = int(right + bias[0])
            top = int(top + bias[1])
            bottom = int(bottom + bias[1])
            stem_box_in_strawberry =np.asarray([top, left, bottom, right])
        return image,boxes[i],stem_box_in_strawberry,strawberry_image,stem_boxes[j]

"""
def get_label_from_imgdataset(image_in):
    #input image type:np
    #output the stem area in strawberry picture
    #choose the first stem(type pil) 可拓展
    image = image_in.copy()
    boxes = yolo.detect_boxes(image)
    # the boxes are set by np.array([None]). Not means the boxes is None.
    if boxes is None:
        return
    #print('strawberry bounding box:')
    #print(boxes)
    x, y = boxes.shape
    return image, boxes
    
    
    for i in range(x):
        strawberry_image = crop_bbox(image, boxes[i])
        # Second stage: find the stem
        stem_boxes = yolo_stage2.detect_boxes(strawberry_image)
        # the boxes are set by np.array([None]). Not means the boxes is None.
        if stem_boxes is None: #stem not detected but strawberry is detected

            return image,boxes[i],None,None,None
        #print('stem bounding box:')
        #print(stem_boxes)
        xt, yt = stem_boxes.shape
        for j in range(xt): #both stem and strawberry is detected
            # get the stem direction
            stem_image = crop_bbox(strawberry_image, stem_boxes[j])
            stem_img_line, w,rho,theta = find_direction(stem_image, 'hough')
            im_paste_left = int(np.maximum(boxes[i][1], 0) + np.maximum(stem_boxes[j][1], 0))
            im_paste_top = int(np.maximum(boxes[i][0], 0) + np.maximum(stem_boxes[j][0], 0))
            image.paste(stem_img_line, (im_paste_left, im_paste_top))
            bias = (np.maximum(boxes[i][1], 0), np.maximum(boxes[i][0], 0))
            image = draw_rectangle(image, stem_boxes[j], bias, w, j)
            draw = ImageDraw.Draw(image)
            rtg = (boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2])
            draw.rectangle(rtg, outline=(30, 144, 255), width=4)
            # 只返回第一个草莓的梗
            top, left, bottom, right = stem_boxes[j]
            left = int(left + bias[0])
            right = int(right + bias[0])
            top = int(top + bias[1])
            bottom = int(bottom + bias[1])
            stem_box_in_strawberry =np.asarray([top, left, bottom, right])
        return image,boxes[i],stem_box_in_strawberry,strawberry_image,stem_boxes[j]

"""
def get_stem_label_from_cropped_dataset(image_in):
    #input image type:np
    #output the stem area in strawberry picture
    #choose the first stem(type pil) 可拓展
    image = image_in.copy()
    boxes = yolo_stage1.detect_boxes(image)
    # the boxes are set by np.array([None]). Not means the boxes is None.
    if boxes is None:
        return
    #print('strawberry bounding box:')
    #print(boxes)
    x, y = boxes.shape
    for i in range(x):
        strawberry_image = image
        # Second stage: find the stem
        stem_boxes = yolo_stage2.detect_boxes(strawberry_image)
        # the boxes are set by np.array([None]). Not means the boxes is None.
        if stem_boxes is None: #stem not detected but strawberry is detected

            return image,boxes[i],None,None,None
        #print('stem bounding box:')
        #print(stem_boxes)
        xt, yt = stem_boxes.shape
        for j in range(xt): #both stem and strawberry is detected
            # get the stem direction
            stem_image = crop_bbox(strawberry_image, stem_boxes[j])
            stem_img_line, w,rho,theta = find_direction(stem_image, 'hough')
            im_paste_left = int(np.maximum(boxes[i][1], 0) + np.maximum(stem_boxes[j][1], 0))
            im_paste_top = int(np.maximum(boxes[i][0], 0) + np.maximum(stem_boxes[j][0], 0))
            image.paste(stem_img_line, (im_paste_left, im_paste_top))
            bias = (np.maximum(boxes[i][1], 0), np.maximum(boxes[i][0], 0))
            image = draw_rectangle(image, stem_boxes[j], bias, w, j)
            draw = ImageDraw.Draw(image)
            rtg = (boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2])
            draw.rectangle(rtg, outline=(30, 144, 255), width=4)
            # 只返回第一个草莓的梗
            top, left, bottom, right = stem_boxes[j]
            left = int(left + bias[0])
            right = int(right + bias[0])
            top = int(top + bias[1])
            bottom = int(bottom + bias[1])
            stem_box_in_strawberry =np.asarray([top, left, bottom, right])
        return image,boxes[i],stem_box_in_strawberry,strawberry_image,stem_boxes[j]

"""
if __name__ == "__main__":
    yolo = YOLO()

    #save_stemimg_dir()

    while True:
        #two open modes
        mode = 1

        if mode == 0:
            img_name = input('Input image filename:')
            assert img_name != 'stop'
            img = 'img/'+img_name+'.jpg'
        elif mode == 1:
            print('Open the image')
            img = tkinter.filedialog.askopenfilename()
            print('Img name:', img)
            assert img != ''

        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            #first stage: find the strawberry
            result= get_pick_img(image)
            if result:
                image,*_ = result
                image.show()
            else:
                image.show()
            

#bug:IMG_5370.JPG,IMG_5718.JPG的stem的框有负数
