import os
import cv2
import glob
import numpy as np
from imutils.object_detection import non_max_suppression

def distance_image(img, color, n_power):
	color_image = img.copy()
	color_image[:] = color
	diff = np.abs(color_image.astype(np.float32) - img.astype(np.float32))/255.
	dist_image = (1 - np.sum(diff**n_power.,axis=-1)**(1./n_power)/3.)*255.
	return dist_image.astype(np.uint8)

def non_max_suppression(boxes, probs=None, scales=None, overlapThresh=0.3):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are integers, convert them to floats -- this
	# is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]

	# compute the area of the bounding boxes and grab the indexes to sort
	# (in the case that no probabilities are provided, simply sort on the
	# bottom-left y-coordinate)
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = y2

	# if probabilities are provided, sort on them instead
	if probs is not None:
		idxs = probs

	# sort the indexes
	idxs = np.argsort(idxs)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the index value
		# to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of the bounding
		# box and the smallest (x, y) coordinates for the end of the bounding
		# box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have overlap greater
		# than the provided overlap threshold
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked
	ret = [boxes[pick].astype("int")]
	if probs is not None:
		ret.append(probs[pick])
	if scales is not None:
		ret.append(scales[pick])
	return ret


BASE_DIR = './data'
VIDEO_IDS_FILE = 'video_ids.txt'
TEMPLETE_FILE = 'template.jpg'
CAPTURE_INTERVAL = 1
FPS = 30
THRESH_DETECT = 0.78


video_low_dir = os.path.join(BASE_DIR, 'video', '360p')
video_high_dir = os.path.join(BASE_DIR, 'video', '1080p')
roi_dir = os.path.join(BASE_DIR, 'video', 'roi')
killed_dir = os.path.join(BASE_DIR, 'killed')

cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.namedWindow('croped', cv2.WINDOW_NORMAL)

template = cv2.imread(TEMPLETE_FILE)
template = distance_image(template, (0,0,255))
template = (np.clip(template - 150, 0, 255) / (255 - 150) * 255).astype(np.uint8)

video_ids = open(VIDEO_IDS_FILE, 'r' ).readlines()
for video_id in video_ids:
	video_id = video_id.strip()
	low_path = os.path.join(video_low_dir, video_id + '.mp4')
	high_path = os.path.join(video_high_dir, video_id + '.mp4')
	roi_path = os.path.join(roi_dir, video_id + '.npz')
	out_dir = os.path.join(killed_dir, video_id)

	if os.path.exists(out_dir): continue

	os.mkdir(out_dir)
	os.mkdir(out_dir + '/detected')
	os.mkdir(out_dir + '/line')

	count = 0
	count_itr = 0
	flag = False
	video_low = cv2.VideoCapture(low_path)
	video_high = cv2.VideoCapture(high_path)
	roi = np.load(roi_path)
	video_low.set(cv2.CAP_PROP_POS_FRAMES, count)
	while True:
		ret, frame = video_low.read()
		if ret == True:
			count += 1
			if count % (FPS*CAPTURE_INTERVAL) == 0:
				h, w, _ = frame.shape
				x1 = int(w*roi['x1'])
				x2 = int(w*roi['x2'])
				y1 = int(h*roi['y1'])
				y2 = int(h*roi['y2'])
				croped = frame[y1:y1+(y2-y1)//4, x1+(x2-x1)//3:x2-(x2-x1)//3]
				cv2.imshow('croped', croped)
				cv2.waitKey(1)

				b, g, r = cv2.split(croped)
				red_pixels = (b < 60) * (g < 40) * (r > 140)
				print('\nvideo id:{0}, frame count:{1}, num red pixels:{2}'.format(video_id, count, np.sum(red_pixels)), end='')

				if np.sum(red_pixels) > 10:
					if not flag: continue

					diff = np.abs(croped.astype(np.int32) - croped_old.astype(np.int32))
					if np.mean(diff) < 20:
						count_itr += 1
					else:
						count_itr = 0
					print(' > mean diff:{0}, consecutive count:{1}'.format(np.mean(diff), count_itr), end='')
					
					if count_itr > 4: continue

					video_high.set(cv2.CAP_PROP_POS_FRAMES, count-1)
					img = video_high.read()[1]

					dist_img = distance_image(img, (0,0,255))
					dist_img = (np.clip(dist_img - 150, 0, 255) / (255 - 150) * 255).astype(np.uint8)

					rects = []
					confidences = []
					scales = []
					for scale in np.linspace(0.7, 1.3, 20)[::-1]:
						scaled = cv2.resize(template.copy(), dsize=None, fx=scale, fy=scale)
						w, h = scaled.shape[::-1]

						res = cv2.matchTemplate(dist_img, scaled, cv2.TM_CCOEFF_NORMED)
						ys, xs = np.where(res > THRESH_DETECT)
						for x, y in zip(xs, ys):
							rects.append([x, y, x + w, y + h])
							confidences.append(res[y,x])
							scales.append(scale)

					if len(confidences) > 0:
						detected = img.copy()

						boxes, probs, scales = non_max_suppression(np.array(rects), probs=np.array(confidences), scales=np.array(scales))
						for bi, ((x1, y1, x2, y2), p, s) in enumerate(zip(boxes, probs, scales)):
							cv2.rectangle(detected, (x1, y1), (x2, y2), (255,255,0), 2)
							cv2.putText(detected, str(p), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
							cv2.putText(detected, str(s), (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
							cv2.imwrite(out_dir + '/line/{0:06d}-{1}.jpg'.format(count, bi), img[y1-5:y2+5])

						print(' > num detected:{0}'.format(len(probs)), end='')
						
						cv2.imwrite(out_dir + '/detected/{0:06d}.jpg'.format(count), detected)
						cv2.imshow('img', detected)
						cv2.waitKey(1)

					else:
						cv2.imshow('img', img)
						cv2.waitKey(1)

				else:
					flag = True
					count_itr = 0

				croped_old = croped
		else:
			break