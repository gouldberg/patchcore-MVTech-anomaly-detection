
import os
import shutil
import glob
import cv2

import PIL
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

import numpy as np
import time


###############################################################################################################
# -------------------------------------------------------------------------------------------------------------
# all test/defective_type into test/ng
# -------------------------------------------------------------------------------------------------------------

base_path = '~/mvtec_ad'

# data_path = os.path.join(base_path, 'mvtec_ad2')
data_path = os.path.join(base_path, 'mpdd')

texture_classes = ["carpet", "grid", "leather", "tile", "wood"]
object_classes = ["cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"]
others = ["bottle"]

# mvtec_class = texture_classes + object_classes + others
mvtec_class = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']

for cls_obj in mvtec_class:
    print(f'processing -- {cls_obj}')
    folder_obj_cls = os.path.join(data_path, cls_obj, 'test')
    folder_obj = os.path.join(data_path, cls_obj, 'test', 'ng')

    if os.path.exists(folder_obj):
        shutil.rmtree(folder_obj)

    os.makedirs(folder_obj)

    folder_list = glob.glob(os.path.join(folder_obj_cls, '*'))
    defective_cls_list = list(set([fold.split('\\')[-1] for fold in folder_list]) - set(['good']) - set(['ng']))

    img_count = 0
    for defective_cls in defective_cls_list:
        print(f'processing -- {cls_obj} - {defective_cls}')
        img_files = glob.glob(os.path.join(folder_obj_cls, defective_cls, '*.png'))

        for img_file in img_files:
            img_count += 1
            img = cv2.imread(img_file)
            save_name = os.path.join(folder_obj, str(img_count).zfill(3) + '.png')
            cv2.imwrite(save_name, img)


##############################################################################################################
# -------------------------------------------------------------------------------------------------------------
# decrease resolution
# -------------------------------------------------------------------------------------------------------------

base_path = '~/mvtec_ad'

# data_path = os.path.join(base_path, 'mvtec_ad_512128')
# data_path = os.path.join(base_path, 'mvtec_ad_512192')
# data_path = os.path.join(base_path, 'kw_448173')
data_path = os.path.join(base_path, 'image_ks\\flexcable')

# texture_classes = ["carpet", "grid", "leather", "tile", "wood"]
# object_classes = ["cable", "capsule", "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper"]
# others = ["bottle"]

# others = ['kwflexfront', 'kwflexleft', 'kwflexright']
others = ['fccase5']

# mvtec_class = texture_classes + object_classes + others
# mvtec_class = ['hazelnut']
mvtec_class = others


for cls_obj in mvtec_class:
    print(f'processing -- {cls_obj}')
    # folder_obj_cls = os.path.join(data_path, cls_obj, 'train\\good')
    # folder_save = os.path.join(data_path, cls_obj, 'train\\good_save')
    folder_obj_cls = os.path.join(data_path, cls_obj, 'test\\good')
    folder_save = os.path.join(data_path, cls_obj, 'test\\good_save')
    # folder_obj_cls = os.path.join(data_path, cls_obj, 'test\\ng')
    # folder_save = os.path.join(data_path, cls_obj, 'test\\ng_save')

    if os.path.exists(folder_save):
        shutil.rmtree(folder_save)

    if not os.path.exists(folder_save):
        os.makedirs(folder_save)

    # img_list = sorted(glob.glob(os.path.join(folder_obj_cls, '*.png')))
    img_list = sorted(glob.glob(os.path.join(folder_obj_cls, '*.jpg')))

    for i in range(len(img_list)):

        img_fname = img_list[i].split('\\')[-1].split('.')[0]
        save_path = os.path.join(folder_save, img_fname + '.png')

        img = cv2.imread(img_list[i])
        # img_resized = cv2.resize(img, dsize=(448, 448))
        # img_resized = cv2.resize(img, dsize=(448, 448))
        img_resized = cv2.resize(img, dsize=(512, 512))
        cv2.imwrite(save_path, img_resized)

    # shutil.rmtree(folder_obj_cls)
    # shutil.move(folder_save, folder_obj_cls)


###############################################################################################################
# -------------------------------------------------------------------------------------------------------------
# transistor rotation
# -------------------------------------------------------------------------------------------------------------

import albumentations as A

base_path = '~/mvtec_ad'

data_path = os.path.join(base_path, 'mvtec_ad2')
# data_path = os.path.join(base_path, 'kw')

# cls_obj = 'kwflexright_224rotate15'
# cls_obj = 'kwflexfront_224rotate15'
# cls_obj = 'kwflexleft_224rotate15'
cls_obj = 'flexcrop3'

folder_obj_cls = os.path.join(data_path, cls_obj, 'train\\good')
folder_save = os.path.join(data_path, cls_obj, 'train\\good_save')
# folder_obj_cls = os.path.join(data_path, cls_obj, 'test\\good')
# folder_save = os.path.join(data_path, cls_obj, 'test\\good_save')
# folder_obj_cls = os.path.join(data_path, cls_obj, 'test\\ng')
# folder_save = os.path.join(data_path, cls_obj, 'test\\ng_save')

if os.path.exists(folder_save):
    shutil.rmtree(folder_save)

if not os.path.exists(folder_save):
    os.makedirs(folder_save)



img_list = glob.glob(os.path.join(folder_obj_cls, '*.png'))
# img_list = sorted(glob.glob(os.path.join(folder_obj_cls, '*.jpg')))

for i in range(len(img_list)):

    img_fname = img_list[i].split('\\')[-1].split('.')[0]
    save_path = os.path.join(folder_save, img_fname + '.png')

    img = PIL.Image.open(img_list[i]).convert("RGB")
    img = transforms.functional.resize(img=img, size=(224, 224))
    # image_rotate = A.Compose([A.augmentations.geometric.rotate.Rotate(limit=15, p=1.0)])(image=np.array(img))['image']
    # image_rotate = A.Compose([A.Rotate([45, -45])])(image=np.array(img))['image']
    image_rotate = A.Rotate([15, 15], p=1)(image=np.array(img))['image']
    # image_rotate = A.Affine(rotate=[90, 90], p=1, mode=cv.BORDER_CONSTANT, fit_output=True)
    image_rotate = cv2.cvtColor(image_rotate, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image_rotate)

    # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # cv2.imwrite(save_path, img)



###############################################################################################################
# -------------------------------------------------------------------------------------------------------------
# data augmentation:  kw flex crop
# -------------------------------------------------------------------------------------------------------------

import albumentations as A

base_path = '~/mvtec_ad'

data_path = os.path.join(base_path, 'kw')

# cls_obj = 'flexcrop2front_aug10rotate15'
# cls_obj = 'flexcrop2left_aug10rotate15'
# cls_obj = 'flexcrop2right_aug10rotate15'

# cls_obj = 'kwflexfront2_aug10rotate15'
# cls_obj = 'kwflexleft2_aug10rotate15'
# cls_obj = 'kwflexright2_aug10rotate15'

# cls_obj = 'flexcrop2front_anime'
# cls_obj = 'flexcrop2left_anime'
cls_obj = 'flexcrop2right_anime'

# folder_obj_cls = os.path.join(data_path, cls_obj, 'train\\good')
# folder_save = os.path.join(data_path, cls_obj, 'train\\good_save')
# folder_obj_cls = os.path.join(data_path, cls_obj, 'test\\good')
# folder_save = os.path.join(data_path, cls_obj, 'test\\good_save')
folder_obj_cls = os.path.join(data_path, cls_obj, 'test\\ng')
folder_save = os.path.join(data_path, cls_obj, 'test\\ng_save')

if os.path.exists(folder_save):
    shutil.rmtree(folder_save)

if not os.path.exists(folder_save):
    os.makedirs(folder_save)


# img_list = glob.glob(os.path.join(folder_obj_cls, '*.png'))
img_list = sorted(glob.glob(os.path.join(folder_obj_cls, '*.jpg')))

aug_cnt = 10
rotate_limit = 15

for i in range(len(img_list)):

    # for j in range(aug_cnt):
    j = 0

    img_fname = img_list[i].split('\\')[-1].split('.')[0]
    save_path = os.path.join(folder_save, img_fname + f'_{str(j).zfill(2)}.png')

    img = PIL.Image.open(img_list[i]).convert("RGB")
    # no resize
    # img = transforms.functional.resize(img=img, size=(224, 224))
    # img = transforms.functional.resize(img=img, size=(448, 448))

    # ----------
    # image_rotate = A.Compose([A.augmentations.geometric.rotate.Rotate(limit=rotate_limit, p=1.0)])(image=np.array(img))['image']
    # image_rotate = A.Compose([A.Rotate([rotate_limit, -rotate_limit])])(image=np.array(img))['image']
    # image_rotate = A.Rotate([rotate_angle, rotate_angle], p=1)(image=np.array(img))['image']
    # image_rotate = A.Affine(rotate=[rotate_angle, rotate_angle], p=1, mode=cv.BORDER_CONSTANT, fit_output=True)
    # image_rotate = cv2.cvtColor(image_rotate, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(save_path, image_rotate)

    # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # cv2.imwrite(save_path, img)

    anime1 = anime_filter1(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    cv2.imwrite(save_path, anime1)


###############################################################################################################
# -------------------------------------------------------------------------------------------------------------
# image processing
# -------------------------------------------------------------------------------------------------------------

import numpy as np
import cv2
import imutils

from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage

def translate(image, x, y):
    # Define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return shifted


def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # Return the rotated image
	return rotated


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def auto_canny(image, sigma=0.33):
    v = np.median(image)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(max(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype('uint8')

    return cv2.LUT(image, table)


def img_proc(img):
    kernel = np.ones((5, 5), np.uint8)
    # img_obj = adjust_gamma(img, gamma=1.75)
    img_obj = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_obj = cv2.GaussianBlur(img_obj, (7, 7), 0)
    # img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_CLOSE, kernel=kernel, iterations=2)
    # img_obj = cv2.morphologyEx(img_obj, cv2.MORPH_OPEN, kernel=kernel, iterations=2)
    # img_obj = cv2.Canny(img_obj, 50, 50)
    return img_obj


def anime_filter1(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    edge = cv2.blur(gray, (3, 3))
    edge = cv2.Canny(edge, 50, 150, apertureSize=3)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    # by pyrMeanShiftFiltering
    img = cv2.pyrMeanShiftFiltering(img, 5, 20)
    return cv2.subtract(img, edge)


def sub_color(src, K):
    Z = src.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((src.shape))


def anime_filter2(img, K):
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    edge = cv2.blur(gray, (3, 3))
    edge = cv2.Canny(edge, 50, 150, apertureSize=3)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    # by k-means
    img = sub_color(img, K)
    return cv2.subtract(img, edge)





base_path = '~/mvtec_ad'

data_path = os.path.join(base_path, 'kw')

cls_obj = 'flexcrop2front_aug10rotate15'
# cls_obj = 'flexcrop2left_aug10rotate15'
# cls_obj = 'flexcrop2right_aug10rotate15'

folder_obj_cls = os.path.join(data_path, cls_obj, 'train\\good')
folder_save = os.path.join(data_path, cls_obj, 'train\\good_save')
# folder_obj_cls = os.path.join(data_path, cls_obj, 'test\\good')
# folder_save = os.path.join(data_path, cls_obj, 'test\\good_save')
# folder_obj_cls = os.path.join(data_path, cls_obj, 'test\\ng')
# folder_save = os.path.join(data_path, cls_obj, 'test\\ng_save')

img_list = glob.glob(os.path.join(folder_obj_cls, '*.png'))
# img_list = sorted(glob.glob(os.path.join(folder_obj_cls, '*.jpg')))

idx = 0

img = cv2.imread(img_list[idx])

# PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).show()


# perform pyramid mean shift filtering to aid the thresholding step
shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
# shifted = cv2.pyrMeanShiftFiltering(img, 11, 21)
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)


# ------------------------------------------------------------------------------------------------------
# sure background
# ------------------------------------------------------------------------------------------------------

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

sure_bg = cv2.dilate(thresh, kernel, iterations=1)


# ------------------------------------------------------------------------------------------------------
# anime filter
# ------------------------------------------------------------------------------------------------------

anime1 = anime_filter1(img)
anime2 = anime_filter2(img, 10)

PIL.Image.fromarray(cv2.cvtColor(anime1, cv2.COLOR_BGR2RGB)).show()
PIL.Image.fromarray(cv2.cvtColor(anime2, cv2.COLOR_BGR2RGB)).show()
PIL.Image.fromarray(cv2.cvtColor(sure_bg, cv2.COLOR_BGR2RGB)).show()


# ----------
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.imshow(labels, cmap="tab20b")
# plt.show()



###############################################################################################################
# -------------------------------------------------------------------------------------------------------------
# data augmentation:  kw sun frare
# -------------------------------------------------------------------------------------------------------------

import albumentations as A

base_path = '~/mvtec_ad'

data_path = os.path.join(base_path, 'kw')

# cls_obj = 'flexcrop2front_aug05sunflare'
# cls_obj = 'flexcrop2left_aug05sunflare'
cls_obj = 'flexcrop2right_aug05sunflare'

folder_obj_cls = os.path.join(data_path, cls_obj, 'train\\good')
folder_save = os.path.join(data_path, cls_obj, 'train\\good_save')
# folder_obj_cls = os.path.join(data_path, cls_obj, 'test\\good')
# folder_save = os.path.join(data_path, cls_obj, 'test\\good_save')
# folder_obj_cls = os.path.join(data_path, cls_obj, 'test\\ng')
# folder_save = os.path.join(data_path, cls_obj, 'test\\ng_save')

if os.path.exists(folder_save):
    shutil.rmtree(folder_save)

if not os.path.exists(folder_save):
    os.makedirs(folder_save)


# img_list = glob.glob(os.path.join(folder_obj_cls, '*.png'))
img_list = sorted(glob.glob(os.path.join(folder_obj_cls, '*.jpg')))

aug_cnt = 5

for i in range(len(img_list)):

    for j in range(aug_cnt):
        # j = 0

        img_fname = img_list[i].split('\\')[-1].split('.')[0]
        save_path = os.path.join(folder_save, img_fname + f'_{str(j).zfill(2)}.png')

        img = PIL.Image.open(img_list[i]).convert("RGB")
        # no resize
        # img = transforms.functional.resize(img=img, size=(224, 224))
        # img = transforms.functional.resize(img=img, size=(448, 448))

        # ----------
        image_aug = A.Compose([A.augmentations.transforms.RandomSunFlare(
            flare_roi=(0.15,0.15,0.25,0.25),
            num_flare_circles_lower=2,
            num_flare_circles_upper=4,
            angle_lower=0.2,
            angle_upper=1,
            src_radius=40*4,
            p=1.0)])(image=np.array(img))['image']
        # ----------
        # image_aug = A.Compose([A.augmentations.transforms.RandomBrightnessContrast(
        #     brightness_limit=0.2, contrast_limit=0.2,
        #     p=1.0)])(image=np.array(img))['image']
        # ----------
        # image_aug = A.Compose([A.augmentations.transforms.ToSepia(
        #     p=1.0)])(image=np.array(img))['image']
        # ----------
        # image_aug = A.Compose([A.augmentations.transforms.HueSaturationValue(
        #     p=1.0)])(image=np.array(img))['image']
        # ----------
        image_aug = cv2.cvtColor(image_aug, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, image_aug)


###############################################################################################################
# -------------------------------------------------------------------------------------------------------------
# rename data file
# -------------------------------------------------------------------------------------------------------------

base_path = '~/mvtec_ad'

data_path = os.path.join(base_path, 'mvtec_ad2')

cls_obj = 'transistor_test15'

# cat0 = 'train'
# cat1 = 'good'

cat0 = 'test'
cat1 = 'good'

# cat0 = 'test'
# cat1 = 'ng'

folder_obj_cls = os.path.join(data_path, cls_obj, f'{cat0}\\{cat1}')
folder_obj = os.path.join(data_path, cls_obj, f'{cat0}\\{cat1}_save')

if os.path.exists(folder_obj):
    shutil.rmtree(folder_obj)

os.makedirs(folder_obj)

img_list = sorted(glob.glob(os.path.join(folder_obj_cls, '*.png')))

for i in range(len(img_list)):
    img = cv2.imread(img_list[i])
    fname = f'{cat0}{cat1}' + img_list[i].split('\\')[-1]
    save_name = os.path.join(folder_obj, fname)
    cv2.imwrite(save_name, img)



###############################################################################################################
# -------------------------------------------------------------------------------------------------------------
# data augmentation:  sun flare parameter tuning
# -------------------------------------------------------------------------------------------------------------

import albumentations as A

base_path = '~/mvtec_ad'

data_path = os.path.join(base_path, 'kw')

cls_obj = 'flexcrop2right_sunflare_param'

folder_obj_cls = os.path.join(data_path, cls_obj, 'train\\good')
folder_save = os.path.join(data_path, cls_obj, 'train\\good_save')
# folder_obj_cls = os.path.join(data_path, cls_obj, 'test\\good')
# folder_save = os.path.join(data_path, cls_obj, 'test\\good_save')
# folder_obj_cls = os.path.join(data_path, cls_obj, 'test\\ng')
# folder_save = os.path.join(data_path, cls_obj, 'test\\ng_save')

# img_list = sorted(glob.glob(os.path.join(folder_obj_cls, '*.png')))
img_list = sorted(glob.glob(os.path.join(folder_obj_cls, '*.jpg')))
print(len(img_list))

# flare_roi = (0, 0, 1, 0.5)
# num_flare_circles_lower = 6
# num_flare_circles_upper = 10
# angle_lower = 0
# angle_upper = 1
# src_radius = 400

# flare_roi = (0.05, 0.05, 0.1, 0.1)
# num_flare_circles_lower = 0
# num_flare_circles_upper = 1
# angle_lower = 0.5
# angle_upper = 1
# src_radius = 40 * 8

flare_roi = (0.05, 0.05, 0.1, 0.1)
num_flare_circles_lower = 0
num_flare_circles_upper = 1
angle_lower = 0.2
angle_upper = 1
src_radius = 40 * 8

image_trans = A.Compose([A.augmentations.transforms.RandomSunFlare(
    flare_roi=flare_roi,
    num_flare_circles_lower=num_flare_circles_lower,
    num_flare_circles_upper=num_flare_circles_upper,
    angle_lower=angle_lower,
    angle_upper=angle_upper,
    src_radius=src_radius,
    p=1.0)])

if os.path.exists(folder_save):
    shutil.rmtree(folder_save)

if not os.path.exists(folder_save):
    os.makedirs(folder_save)

idx = 0
aug_cnt = 5
for j in range(aug_cnt):
    img = PIL.Image.open(img_list[idx]).convert("RGB")

    img_fname = img_list[idx].split('\\')[-1].split('.')[0]
    save_path = os.path.join(folder_save, img_fname + f'{str(j).zfill(2)}_{flare_roi}_{num_flare_circles_lower}_{num_flare_circles_upper}_{str(int(angle_lower*10)).zfill(3)}_{str(int(angle_upper*10)).zfill(3)}_{str(src_radius).zfill(3)}.png')

    img = PIL.Image.open(img_list[idx]).convert("RGB")
    image_sunflare = image_trans(image=np.array(img))['image']
    image_sunflare = cv2.cvtColor(image_sunflare, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, image_sunflare)


