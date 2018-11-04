from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage
import face_alignment
from skimage import io

# # Create your API key in your account's `Manage your API keys` page:
# # https://clarifai.com/developer/account/keys

# app = ClarifaiApp(api_key='1a0567bf3af94b23a7f4f5a872df2c06')

# # # You can also create an environment variable called `CLARIFAI_API_KEY` 
# # # and set its value to your API key.
# # # In this case, the construction of the object requires no `api_key` argument.

 
# print("________")
# model = app.models.get('face-v1.3')
# image = ClImage(filename='./test.png')
# response = model.predict([image])
# print(response)


# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cpu')

# input = io.imread('./test.png')
# preds = fa.get_landmarks(input)
# print(preds)
import stasm
import sys
import os
import dlib
import glob
import matplotlib.pyplot as plt
import skimage.io as skio
import cv2

import numpy as np

img = cv2.imread('test.jpeg')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

# img = cv2.imread('test.jpeg')
# mask = np.zeros(img.shape[:2],np.uint8)
# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)
# rect = (50,50,450,290)
# cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
# mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# img = img*mask2[:,:,np.newaxis]
# plt.imshow(img),plt.colorbar(),plt.show()

# if len(sys.argv) != 2:
#     exit()


# faces_folder_path = sys.argv[1]

# predictor_path = os.path.join(sys.path[0], 'predictor.dat')
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(predictor_path)

# for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
#     #print("Processing file: {}".format(f))
#     img = dlib.load_rgb_image(f)
#     im1 = skio.imread(f)
#     plt.imshow(im1)
#     plt.show()
	

    
#     # Ask the detector to find the bounding boxes of each face. The 1 in the
#     # second argument indicates that we should upsample the image 1 time. This
#     # will make everything bigger and allow us to detect more faces.
#     dets = detector(img, 1)
#     print(dets)
#     print("Number of faces detected: {}".format(len(dets)))
#     for k, d in enumerate(dets):
#         print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#             k, d.left(), d.top(), d.right(), d.bottom()))
#         # Get the landmarks/parts for the face in box d.
#         shape = predictor(img, d)
#         # points=stasm.search_single(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
#         # print(points)
#         #0-17 脸
#         #18-22 左眉毛
#         #23-27 右眉毛
#         #28-36 鼻子
#         #37-42 左眼睛
#         #43-48 右眼睛
#         #49-68 嘴 
#         for i in shape.parts():
#         	x, y = i.x, i.y
#         	plt.scatter(x, y, s=5, c='red', marker='o')
#         plt.show() 
#         # for i in points:
#         #     x, y = i
#         #     plt.scatter(x, y, s=5, c='red', marker='o')
#         # plt.show() 
#         # Draw the face landmarks on the screen.

#     dlib.hit_enter_to_continue()
