import os
from tqdm import tqdm
from pytube import YouTube

BASE_DIR = './data'
VIDEO_IDS_FILE = 'video_ids.txt'

video_dir = os.path.join(BASE_DIR, 'video')
video_ids = open(VIDEO_IDS_FILE, 'r' ).readlines()

resolutions = ['360p', '1080p']
for res in resolutions:
	res_dir = os.path.join(video_dir, res)
	if not os.path.exists(res_dir): os.makedirs(res_dir)

pbar = tqdm(video_ids)
for video_id in pbar:
	video_id = video_id.strip()
	#if os.path.exists(os.path.join(video_dir, resolutions[0], video_id + '.mp4')): continue
	for res in resolutions:
		pbar.set_description('{0} {1}'.format(video_id, res))

		res_dir = os.path.join(video_dir, res)
		if os.path.exists(os.path.join(res_dir, video_id + '.mp4')): continue
		
		yt = YouTube('https://www.youtube.com/watch?v='+video_id)
		try:
			video = yt.streams.filter(mime_type='video/mp4', resolution=res).first()
			video.download(res_dir, filename=video_id)
		except:
			print('skip:', video_id, res)
			continue

		