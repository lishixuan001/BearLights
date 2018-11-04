from docopt import docopt
import os
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import dlib
import scipy.spatial as spatial
from builtins import range
from skimage import feature
import io
from google.cloud import vision
from scipy.misc import imsave
from PIL import Image
import base64

def triangular_affine_matrices(vertices, src_points, dest_points):
	ones = [1, 1, 1]
	for tri_indices in vertices:
		src_tri = np.vstack((src_points[tri_indices, :].T, ones))
		dst_tri = np.vstack((dest_points[tri_indices, :].T, ones))
		mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
		yield mat

def weighted_average(img1, img2, percent=0):
	if percent <= 0:
		return img2
	elif percent >= 1:
		return img1
	else:
		return cv2.addWeighted(img1, percent, img2, 1-percent, 0)

def weighted_average_points(start_points, end_points, percent=0.5):
	if percent <= 0:
		return end_points
	elif percent >= 1:
		return start_points
	else:
		return np.asarray(start_points*percent + end_points*(1-percent), np.int32)

def morph(src_img, src_points, dest_img, dest_points, width=500, height=600, background='black'):
	size = (width, height)
	points = weighted_average_points(src_points, dest_points, 1)

	src_face = warp_image(src_img, src_points, points, size)
	end_face = warp_image(dest_img, dest_points, points, size)
	average_face = weighted_average(src_face, end_face, 0.5)[:,:,::-1]
	# gray_image = cv2.cvtColor(average_face, cv2.COLOR_BGR2GRAY)
	# edges = feature.canny(gray_image)
	cv2.imwrite('messigray.png',average_face[:,:,::-1])
	return average_face, points

#def ranking(src_img, src_points, dest_img, dest_points):


def positive_cap(num):
	if num < 0:
		return 0, abs(num)
	else:
		return num, 0

def roi_coordinates(rect, size, scale):
	rectx, recty, rectw, recth = rect
	new_height, new_width = size
	mid_x = int((rectx + rectw/2) * scale)
	mid_y = int((recty + recth/2) * scale)
	roi_x = mid_x - int(new_width/2)
	roi_y = mid_y - int(new_height/2)

	roi_x, border_x = positive_cap(roi_x)
	roi_y, border_y = positive_cap(roi_y)
	return roi_x, roi_y, border_x, border_y

def scaling_factor(rect, size):
	new_height, new_width = size
	rect_h, rect_w = rect[2:]
	height_ratio = rect_h / new_height
	width_ratio = rect_w / new_width
	scale = 1
	if height_ratio > width_ratio:
		new_recth = 0.8 * new_height
		scale = new_recth / rect_h
	else:
		new_rectw = 0.8 * new_width
		scale = new_rectw / rect_w
	return scale

def resize_image(img, scale):
	
	cur_height, cur_width = img.shape[:2]
	new_scaled_height = int(scale * cur_height)
	new_scaled_width = int(scale * cur_width)

	return cv2.resize(img, (new_scaled_width, new_scaled_height))

def resize_align(img, points, size):
	new_height, new_width = size
	rect = cv2.boundingRect(np.array([points], np.int32))
	scale = scaling_factor(rect, size)
	img = resize_image(img, scale)
	cur_height, cur_width = img.shape[:2]
	roi_x, roi_y, border_x, border_y = roi_coordinates(rect, size, scale)
	roi_h = np.min([new_height-border_y, cur_height-roi_y])
	roi_w = np.min([new_width-border_x, cur_width-roi_x])
	crop = np.zeros((new_height, new_width, 3), img.dtype)
	crop[border_y:border_y+roi_h, border_x:border_x+roi_w] = (
		 img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w])
	points[:, 0] = (points[:, 0] * scale) + (border_x - roi_x)
	points[:, 1] = (points[:, 1] * scale) + (border_y - roi_y)
	return (crop, points)

def boundary_points(points):
	x, y, w, h = cv2.boundingRect(points)
	buffer_percent = 0.1
	spacerw = int(w * buffer_percent)
	spacerh = int(h * buffer_percent)
	return [[x+spacerw, y+spacerh],
					[x+w-spacerw, y+spacerh]]

def face_points(img, add_boundary_points=True):
	predictor_path = os.path.join(sys.path[0], 'predictor.dat')
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(predictor_path)
	try:
		dets = detector(img, 1)
		shape = predictor(img, dets[0])
	except Exception as e:
		print('Failed finding face points: ', e)
		return []

	points = []
	for i in shape.parts():
		points.append([i.x, i.y])
	points=np.array(points)
	points = points.astype(np.int32)
	if len(points) == 0:
		return points

	if add_boundary_points:
		return np.vstack([points, boundary_points(points)])

	return points

def warp_image(src_img, src_points, dest_points, dest_shape, dtype=np.uint8):
	num_chans = 3
	src_img = src_img[:, :, :3]

	rows, cols = dest_shape[:2]
	result_img = np.zeros((rows, cols, num_chans), dtype)

	delaunay = spatial.Delaunay(dest_points)
	tri_affines = np.asarray(list(triangular_affine_matrices(
		delaunay.simplices, src_points, dest_points)))

	process_warp(src_img, result_img, tri_affines, dest_points, delaunay)

	return result_img

def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
	roi_coords = grid_coordinates(dst_points)
	roi_tri_indices = delaunay.find_simplex(roi_coords)

	for simplex_index in range(len(delaunay.simplices)):
		coords = roi_coords[roi_tri_indices == simplex_index]
		num_coords = len(coords)
		out_coords = np.dot(tri_affines[simplex_index],
												np.vstack((coords.T, np.ones(num_coords))))
		x, y = coords.T
		result_img[y, x] = bilinear_interpolate(src_img, out_coords)

	return None

def bilinear_interpolate(img, coords):
	int_coords = np.int32(coords)
	x0, y0 = int_coords
	dx, dy = coords - int_coords

	# 4 Neighour pixels
	q11 = img[y0, x0]
	q21 = img[y0, x0+1]
	q12 = img[y0+1, x0]
	q22 = img[y0+1, x0+1]

	btm = q21.T * dx + q11.T * (1 - dx)
	top = q22.T * dx + q12.T * (1 - dx)
	inter_pixel = top * dy + btm * (1 - dy)

	return inter_pixel.T

def grid_coordinates(points):
	xmin = np.min(points[:, 0])
	xmax = np.max(points[:, 0]) + 1
	ymin = np.min(points[:, 1])
	ymax = np.max(points[:, 1]) + 1
	return np.asarray([(x, y) for y in range(ymin, ymax)
										 for x in range(xmin, xmax)], np.uint32)

def morpher(source, dest, background='black'):
	width, height=500, 600
	size=(width, height)
	img = cv2.imread(os.path.join(sys.path[0], source))
	points = face_points(img)
	if len(points) == 0:
		print('No face in source')
		return None
	src_img, src_points=resize_align(img, points, size)

	img = cv2.imread(os.path.join(sys.path[0], dest))
	points = face_points(img)
	if len(points) == 0:
		print('No face in dest')
		return None

	dest_img, dest_points=resize_align(img, points, size)
	return morph(src_img, src_points, dest_img, dest_points, width, height, background)

def grade(combined_face, combined_points, finished_face):
	width, height=500, 600
	size=(width, height)
	img = cv2.imread(os.path.join(sys.path[0], finished_face))
	points = face_points(img)
	if len(points) == 0:
		print('No face in source')
		return None
	src_img, src_points=resize_align(img, points, size)
	return grade_by_pix(src_img, src_points, combined_face, combined_points, width, height)


def grade_by_pix(src_img, src_points, combined_face, combined_points, width, height):
	size=(width, height)
	points = combined_points
	src_img=src_img[:,:,::-1]
	src_img=cv2.normalize(src_img, None, 0,255,cv2.NORM_MINMAX)
	combined_face=cv2.normalize(combined_face, None, 0,255,cv2.NORM_MINMAX)
	#plt.imshow(combined_face)
	#plt.show()
	src_face = warp_image(src_img, src_points, combined_points, size)
	# gray_image = cv2.cvtColor(src_face, cv2.COLOR_RGB2GRAY)
	# edges = feature.canny(gray_image)
	#plt.imshow(src_face)
	#plt.show()
	results=[]
	results1=[]
	# # left_eye=np.hstack([points[17:22],points[36:40]])
	# # right_eye=np.hstack([points[22:27],points[43:47]])
	# # right_eye=np.array([[points[i][0] for i in range(22, 27)] + [points[i][0] for i in range(42, 46)], [points[i][1] for i in range(22, 27)] + [points[i][1] for i in range(42, 46)]])
	right_eye = []
	for i in range(22, 27):
		right_eye.append(list(points[i]))
	for i in range(43, 48):
		right_eye.append(list(points[i]))
	right_eye=np.array(right_eye).reshape((len(right_eye), 2))

	right_crop = warp_image(src_face, right_eye, right_eye, size)
	right_crop_comb = warp_image(combined_face, right_eye, right_eye, size)
	right_crop_file = io.BytesIO()
	imsave(right_crop_file, right_crop, format='png')
	right_crop_comb = warp_image(combined_face, right_eye, right_eye, size)
	right_crop_comb_file = io.BytesIO()
	imsave(right_crop_comb_file, right_crop_comb, format='png')
	#plt.imshow(right_crop)
	#plt.show()
	#plt.imshow(right_crop_comb)
	#plt.show()
	result = get_grade(right_crop_file, right_crop_comb_file)
	right_crop_file.close()
	right_crop_comb_file.close()
	results.append(result)
	results1.append(get_image_difference(right_crop, right_crop_comb))

	left_eye=[]
	for i in range(17, 22):
		left_eye.append(points[i])
	for i in range(36, 40):
		left_eye.append(points[i])
	left_eye=np.array(left_eye).reshape((len(left_eye), 2))
	left_crop = warp_image(src_face, left_eye, left_eye, size)
	left_crop_comb = warp_image(combined_face, left_eye, left_eye, size)
	left_crop_file = io.BytesIO()
	imsave(left_crop_file, left_crop, format='png')
	left_crop_comb = warp_image(combined_face, left_eye, left_eye, size)
	left_crop_comb_file = io.BytesIO()
	imsave(left_crop_comb_file, left_crop_comb, format='png')
	#plt.imshow(left_crop)
	#plt.show()
	#plt.imshow(left_crop_comb)
	#plt.show()
	left_eye_bound=boundary_points(np.array(left_eye))
	result = get_grade(left_crop_file, left_crop_comb_file)
	left_crop_file.close()
	left_crop_comb_file.close()
	results.append(result)
	results1.append(get_image_difference(left_crop, left_crop_comb))

	nose=[]
	for i in range(30, 36):
		nose.append(points[i])
	nose=np.array(nose).reshape((len(nose), 2))
	nose_crop = warp_image(src_face, nose, nose, size)
	nose_crop_comb = warp_image(combined_face, nose, nose, size)
	nose_crop_file = io.BytesIO()
	imsave(nose_crop_file, nose_crop, format='png')
	nose_crop_comb = warp_image(combined_face, nose, nose, size)
	nose_crop_comb_file = io.BytesIO()
	imsave(nose_crop_comb_file, nose_crop_comb, format='png')
	#plt.imshow(nose_crop)
	#plt.show()
	#plt.imshow(nose_crop_comb)
	#plt.show()
	nose_bound=boundary_points(np.array(nose))
	result = get_grade(nose_crop_file, nose_crop_comb_file)
	nose_crop_file.close()
	nose_crop_comb_file.close()
	results.append(result)
	results1.append(get_image_difference(nose_crop, nose_crop_comb))


	mouth=[]
	for i in range(48, 68):
		mouth.append(points[i])
	mouth=np.array(mouth).reshape((len(mouth), 2))
	mouth_crop = warp_image(src_face, mouth, mouth, size)
	mouth_crop_file = io.BytesIO()
	imsave(mouth_crop_file, mouth_crop, format='png')
	mouth_crop_comb = warp_image(combined_face, mouth, mouth, size)
	mouth_crop_comb_file = io.BytesIO()
	imsave(mouth_crop_comb_file, mouth_crop_comb, format='png')
	#plt.imshow(mouth_crop)
	#plt.show()
	#plt.imshow(mouth_crop_comb)
	#plt.show()
	mouth_bound=boundary_points(np.array(mouth))
	result = get_grade(mouth_crop_file, mouth_crop_comb_file)
	mouth_crop_file.close()
	mouth_crop_comb_file.close()
	results.append(result)
	results1.append(get_image_difference(mouth_crop, mouth_crop_comb))
	return results, results1

def detect_properties(path):
    """Detects image properties in the file."""
    client = vision.ImageAnnotatorClient()

    # with Image.open(path) as image_file:
    # 	content = image_file.tobytes().read()

    image = vision.types.Image(content=path.getvalue())

    response = client.image_properties(image=image)
    props = response.image_properties_annotation
    # print('Properties:')

    # for color in props.dominant_colors.colors:
    #     print('fraction: {}'.format(color.pixel_fraction))
    #     print('\tr: {}'.format(color.color.red))
    #     print('\tg: {}'.format(color.color.green))
    #     print('\tb: {}'.format(color.color.blue))
    #     print('\ta: {}'.format(color.color.alpha))

    return [[color.pixel_fraction, color.color.red, color.color.green, color.color.blue] for color in props.dominant_colors.colors]

def get_grade(path1, path2):
    """
        Grades the image at path2 based on the differences in properties of the image at path1
    """
    color_properties_1 = detect_properties(path1)
    color_properties_1.sort(key=lambda x: x[0])
    color_properties_2 = detect_properties(path2)
    color_properties_2.sort(key=lambda x: x[0])
    # while len(color_properties_2) != len(color_properties_1):
    #     if len(color_properties_1) > len(color_properties_2):
    #         color_properties_2 += [0, 0, 0, 0]
    #     elif len(color_properties_2) > len(color_properties_1):
    #         color_properties_1 += [0, 0, 0, 0]

    return 1 - np.linalg.norm(np.array(color_properties_1[-2])-np.array(color_properties_2[-2]))/441.672956

def get_image_difference(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def main():
	source, dest = 'pic1.jpg', 'pic3.jpg'
	face, points=morpher(source, dest)
	# plt.imshow(face)
	# plt.show()
	finished = 'pic3.jpg'
	res, score = grade(face, points, finished)
	if score[0]*(1.7-res[0])<= 150:
		print("Right eye is good")
	else:
		print("Not done with right eye")
	if score[1]*(1.7-res[1])<= 220:
		print("Left eye is good")
	else:
		print("Not done with left eye")
	if score[2]*(1.9-res[2])<= 150:
		print("Nose is good")
	else:
		print("Not done with nose")
	if score[3]*(1.9-res[3])<= 80:
		print("Mouth is good")
	else:
		print("Not done with mouth")


if __name__ == "__main__":
	main()
