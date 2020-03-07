import numpy as np
import cv2
import cvui
import glob
import argparse
import os
import re

def param(frame, x, y, width, value, min, max, name, step=1):
	cvui.text(frame, x, y, name)
	cvui.counter(frame, x, y+13, value, step, "%.0Lf")
	cvui.trackbar(frame, x+100, y, width, value, min, max, 0, '%.0Lf', cvui.TRACKBAR_DISCRETE, step)
	value[0] = np.clip(value[0], min, max).astype(type(value[0]))
	
def anotate(img, out_path, wnd_height = 800):
	img_height, img_width = img.shape[:2]
	wnd_width = int(wnd_height*img_width/img_height)

	frame = np.zeros((wnd_height+200,wnd_width,3), np.uint8)
	roi_size = [img_height//10*5]
	roi_shiftx = [0]
	roi_shifty = [0]
	x1 = [0]
	x2 = [img_width//2]
	y1 = [0]
	y2 = [img_height//2]

	WINDOW_NAME = 'Roi Anotater'
	cvui.init(WINDOW_NAME)

	anchor = cvui.Point()

	while(True):
		
		y = cvui.mouse().y
		x = cvui.mouse().x
		
		if (x>0 and x<wnd_width and y>0 and y<wnd_height):
			if cvui.mouse(cvui.DOWN):
				anchor.y = y
				anchor.x = x
				
			if cvui.mouse(cvui.IS_DOWN):
				if x > anchor.x and y > anchor.y:
					x1[0] = anchor.x*img_width//wnd_width
					x2[0] = x*img_width//wnd_width
					y1[0] = anchor.y*img_height//wnd_height
					y2[0] = y*img_height//wnd_height
					

		frame[:] = (49, 52, 49)
		
		img_tmp = img.copy()
		img_tmp = cv2.rectangle(img_tmp, (x1[0], y1[0]), (x2[0], y2[0]), (0,0,255), 1)
		cvui.image(frame, 0, 0, cv2.resize(img_tmp, (wnd_width,wnd_height)))
		param(frame, 50, wnd_height+ 10, wnd_width-400, x1,  0, x2[0], 'x1', 1)
		param(frame, 50, wnd_height+ 50, wnd_width-400, x2, x1[0], img_width, 'x2', 1)
		param(frame, 50, wnd_height+ 90, wnd_width-400, y1,  0, y2[0], 'y1', 1)
		param(frame, 50, wnd_height+130, wnd_width-400, y2, y1[0], img_height, 'y2', 1)
		 
		cvui.update()
		cv2.imshow(WINDOW_NAME, frame)
		
		if cv2.waitKey(20) == 27:
			np.savez(out_path, x1=x1[0]/img_width, x2=x2[0]/img_width, y1=y1[0]/img_height, y2=y2[0]/img_height)
			break

BASE_DIR = './data'
VIDEO_IDS_FILE = 'video_ids.txt'

video_dir = os.path.join(BASE_DIR, 'video', '360p')
roi_dir = os.path.join(BASE_DIR, 'video', 'roi')

videoIDs = open(VIDEO_IDS_FILE, 'r' ).readlines()
for videoID in videoIDs:
	videoID = videoID.strip()

	video_path = os.path.join(video_dir, videoID + '.mp4')
	roi_path = os.path.join(roi_dir, videoID + '.npz')

	if os.path.exists(roi_path): continue

	video = cv2.VideoCapture(video_path)
	half = video.get(cv2.CAP_PROP_FRAME_COUNT)//2
	video.set(cv2.CAP_PROP_POS_FRAMES, half)
	ret, frame = video.read()
	if ret == True:
		print(videoID)
		anotate(frame, roi_path)
		
cv2.destroyAllWindows()
