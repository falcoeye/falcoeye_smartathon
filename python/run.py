
import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import pytesseract
import torchvision.transforms as T

from timm.models import load_checkpoint
from bench import DetBenchPredict
from model import EfficientDet
from model_config import get_efficientdet_config

import logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
# Dhahran
#GPS_X1, GPS_Y1, GPS_X2, GPS_Y2 = 1570,1853,2382,1950 
# Riyadh
GPS_X1, GPS_Y1, GPS_X2, GPS_Y2 = 1560,1853,2410,1950 
GLASS_LINE_Y = 1350

getthresholds = {'d0' : [0.100,0.100,0.100,0.100],'d0aug' : [0.223,0.21,0.23,0.213],
				'd1' : [0.312,0.227,0.321,0.291],'d1aug' : [0.249,0.216,0.245,0.237],
				'd2' : [0.316,0.234,0.339,0.298],'d2aug' : [0.286,0.257,0.327,0.301],
				'd3' : [0.385,0.318,0.436,0.384],'d3aug' : [0.328,0.375,0.411,0.342],
				'd4' : [0.388,0.322,0.399,0.391],'d7' : [0.353,0.277,0.378,0.368]}

def mkdir(path):
	try: 
		os.mkdir(path) 
	except OSError as error: 
		pass 

def get_gps_value(img):
	def deg_to_dec(deg,letter):
		d,m = deg.split()
		dc = float(d) + float(m.strip(letter))/60
		return dc
	possibl_mistakes = {
		"°":" "," N":"N"," E":"E",
		"'":"","°":" ",". ":"."," .":".",
		"~": "","-—": "","— ":"",
		"£":"E",") ":"","| ":"","/":"",
		"~-":""
	}
	gps_part = img[GPS_Y1:GPS_Y2,GPS_X1:GPS_X2].copy()
	# threshold to keep only white
	gps_part = (250 - np.clip(gps_part,250,255))
	gps_part = Image.fromarray(gps_part).convert('L')
	gps_value = pytesseract.image_to_string(gps_part)
	for k,v in possibl_mistakes.items():
		gps_value = gps_value.strip().replace(k,v)
		
	try:	
		east,north = gps_value.split(",")
		north,east = deg_to_dec(north.strip(),"N"),deg_to_dec(east.strip(),"E")
		return north,east
	except:
		logging.warning(f"Couldn't parse GPS Value: {gps_value}")
		return None,None

def drawonimage(image,boxes,th):
	labels_colors = {
		1: ("D00",(0,255,255)),
		2: ("D10",(0,0,255)),
		3: ("D20",(255,120,255)),
		4: ("D40",(255,0,255))
	}
	for item in boxes:
		label,color = labels_colors[item["category_id"]]
		(x,y),(x2,y2) = (int(item['bbox'][0]),int(item['bbox'][1])), (int((item['bbox'][0]+item['bbox'][2])), int(item['bbox'][1]+item['bbox'][3])+30)
		image = cv2.rectangle(image, (x,y), (x2, y2), color, 2)
		image = cv2.putText(image,  str(label),(int(item['bbox'][0]),int(item['bbox'][1])), cv2.FONT_HERSHEY_SIMPLEX , 0.7, color, 2, cv2.LINE_AA) 
	

	return image
		
def process_detections(img,detections,
	threshold,draw=False):
	results = []
	final_image = img.copy()
	for det in detections:
		score = float(det[4])
		if score < 0.01:  # stop when below this threshold, scores in descending order
			break

		category_id = int(det[5])
		category_id = min(4,category_id)
		cat_threshold = threshold[category_id-1]
		box = det[0:4].tolist()
		_,y,width,height = box

		if category_id<=0 or score < cat_threshold or y > GLASS_LINE_Y:
			continue
	
		length = ((width**2 + height**2)**0.5)
		
		coco_det = dict(
			bbox=box,
			score=score,
			category_id=category_id,
			length=length
		)
		results.append(coco_det)

	if draw:
		final_image = drawonimage(final_image,results,threshold) 
	return results,final_image

class ResizePad:
	def __init__(self,target_size: int, fill_color: tuple = (0, 0, 0)):
		self.target_size = target_size,target_size
		self.fill_color = fill_color

	def __call__(self, img,scale):
		new_img = Image.new("RGB", (self.target_size[1], self.target_size[0]), color=self.fill_color)
		width, height = img.size
		scaled_h = int(height * scale)
		scaled_w = int(width * scale)
		img = img.resize((scaled_w, scaled_h), Image.BILINEAR)
		new_img.paste(img)
		return new_img

class RoadDamage:
	"""
	A custom model handler implementation.
	"""
	
	def __init__(self,model_backbone="tf_efficientdet_d2"):
		self.explain = False
		self.target = 0
		self.resizer = None
		self.tensorizer = T.ToTensor()
		self.normalizer = T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
		self.inputlayersize = None
		self.model_backbone = model_backbone

	def calculate_scale(self,width,height):
		img_scale_y = self.inputlayersize / height
		img_scale_x = self.inputlayersize / width
		img_scale = min(img_scale_y, img_scale_x)
		return img_scale

	def initialize(self):
		"""
		Initialize model. This will be called during model loading time
		:return:
		"""
		self.map_location = "cpu"
		self.device = torch.device(
			self.map_location
		)
		
		self.model = self._load_pickled_model()
		self.model.to(self.device)
		self.model.eval()
		
	def _load_pickled_model(self):
		config = get_efficientdet_config(self.model_backbone)
		model = EfficientDet(config)
		load_checkpoint(model,args.checkpoint)
		self.inputlayersize = config["image_size"]
		self.resizer = ResizePad(self.inputlayersize)
		model = DetBenchPredict(model,config)
		return model
	
	def predict(self,data):
		imgs,scales,input_sizes = data
		return self.model(imgs,scales,input_sizes)	
	
	def preprocess(self, data):
		"""The preprocess function of MNIST program converts the input data to a float tensor

		Args:
			data (List): Input data from the request is in the form of a Tensor

		Returns:
			list : The preprocess function returns the input image as a list of float tensors.
		"""
		img = Image.fromarray(data)
		width, height = img.size

		scale = self.calculate_scale(width, height)
		img_tensor = self.normalizer(self.tensorizer(self.resizer(img,scale)))
		
		images = [img_tensor]
		scales = [1/scale] # to recover
		sizes = [[width,height]]

		images_tensor = torch.stack(images).to(self.device)
		scales_tensor = torch.Tensor(scales).to(self.device)
		sizes_tensor = torch.Tensor(sizes).to(self.device)
		return images_tensor,scales_tensor,sizes_tensor
	
	def postprocess(self, outputs):
		"""
		Return inference result.
		:param inference_output: list of inference output
		:return: list of predict results
		"""
		# Take output from network and post-process to desired format
		logger.info(f"Postprocess: calculating response")
		response = outputs.detach().cpu().numpy().astype(np.float32)
		return response.reshape(-1,6)

class FileSource:
	def __init__(self, filename):
		self._filename = filename
		self._reader = None
		self.width = -1
		self.height = -1
		self.frames_per_second = -1
		self.num_frames = -1
		self._current_frame = 0

	def open(self):
		self._reader = cv2.VideoCapture(self._filename)
		self._reader.set(cv2.CAP_PROP_FPS, 25)
		self.width = int(self._reader.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.height = int(self._reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.diameter = (self.width**2+self.height**2)**0.5
		self.frames_per_second = self._reader.get(cv2.CAP_PROP_FPS)
		self.num_frames = int(self._reader.get(cv2.CAP_PROP_FRAME_COUNT))
		logging.info(f"Opening video with {self.num_frames}, frame rate {self.frames_per_second}, and  Video diamter {self.diameter}")
	def read(self):
		read,frame = self._reader.read()
		self._current_frame += 1
		return read,frame
	
	def close(self):
		self._reader.release()
		self.width = -1
		self.height = -1
		self.frames_per_second = -1
		self.num_frames = -1

class FileSink:
	def __init__(self, filename):
		self._filename = filename
		self._writer = None

	def open(self, frames_per_second, width, height):
		self._writer = cv2.VideoWriter(
			self._filename,
			fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
			fps=30,
			frameSize=(width, height),
			isColor=True,
		)

	def sink(self, frame):

		self._writer.write(frame)

	def close(self):
		self._writer.release()

def get_args_parser():
	parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
	
	parser.add_argument('--file', default='',
						help='file to predict')
	parser.add_argument('--checkpoint', default='',
						help='path where the trained weights saved')
	parser.add_argument('--backbone', default='tf_efficientdet_d0',
						help='Model backbone tf_efficientdet_d[0-7]')
	parser.add_argument('--sample_every', default=10,type=int,
						help='Sample a frame for prediction every')
	parser.add_argument('--draw', action="store_true",help="Draw bounding boxes")
	parser.add_argument('--video', action="store_true",help="Write sampled frames in a video")
	parser.add_argument('--output', type=str,help="File to write data to")
	parser.add_argument('--nogps', action="store_true",help="Draw bounding boxes")
	parser.add_argument('--cracks_imgs', action="store_true",help="Output cracks images")

	
	return parser

def main_video(args):
	sample_every = args.sample_every
	source = FileSource(args.file)
	filename = os.path.basename(args.file)
	source.open()
	if args.video:
		sink = FileSink(args.file.replace(".mp4","_pred.mp4"))
		sink.open(source.frames_per_second,source.width,source.height)
	
	model = RoadDamage(args.backbone)
	model.initialize()
	
	threshold = getthresholds[args.checkpoint.split("/")[-1].split("_")[0]]

	output_dir = args.output
	mkdir(output_dir)
	if os.path.exists(f"{output_dir}/data.csv"):
		with open(f"{output_dir}/data.csv") as f:
			data = f.read().split("\n")
	else:
		data = ["file,frame,x1,y2,x2,y2,latitude,longitude,category,length,normalized_length"]

	read, frame = source.read()
	count = 0
	
	mkdir(f"{output_dir}/imgs/")
	while read:
		count += 1
	
		if count % sample_every != 0:
			# Read next frame and skip
			read, frame = source.read()
			continue
			
		im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		north,east = None,None
		if not args.nogps:
			# Read gps value
			north,east = get_gps_value(im_rgb)
			if north is None or east is None:
				logging.warning(f"Couldn't read gps value for frame {count}")
			else:
				north,east = round(north,6),round(east,6)
			
		# Find road cracks
		img_tensor = model.preprocess(im_rgb)
		predicted = model.predict(img_tensor)[0]
		final_det,final_img = process_detections(im_rgb,predicted,
			threshold,draw=args.draw) 

		if args.video: 
			# Sink to output
			final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
			sink.sink(final_img)

		# Outputting
		n_crack = len(final_det)
		total_length = 0
		n_cat = [0]*4
		record = None
		for index,d in enumerate(final_det):
			length = d["length"]
			n_length = length/source.diameter
			category = d["category_id"]
			if args.cracks_imgs:
				x1,y1,width,height = [int(i) for i in d["bbox"]]
				x2,y2 = x1+width,y1+height
				crack_part = Image.fromarray(im_rgb[y1:y2,x1:x2].copy())
				crack_part.save(f"{output_dir}/imgs/{count}_{index}_{category}.jpg")
			record = f"{filename},{count},{x1},{y1},{x2},{y2},{north},{east},{category},{round(length,3)},{round(n_length,3)}"
			data.append(record)

		if not record:
			logging.info(f"{count}/{source.num_frames}")
		else:
			logging.info(f"{count}/{source.num_frames} {record}")

		# Write data
		with open(f"{output_dir}/data.csv","w") as f:
			f.write("\n".join(data))

			
		
		# Read next frame
		read, frame = source.read()

	if args.video:
		sink.close()
	source.close()

def main_image(args):
	model = RoadDamage(args.backbone)
	img = cv2.imread(args.file)
	source = img[:,:,:3]
	sink = args.file.replace(".jpg","_pred.jpg").replace(".png","_pred.jpg")
	model.initialize()
	north,east = get_gps_value(source)
	logging.info(f"{north},{east}")
	img_tensor = model.preprocess(source)
	predicted = model.predict(img_tensor)[0]
	threshold = getthresholds[args.checkpoint.split("/")[-1].split("_")[0]]
	final_det,final_img = process_detections(source,predicted,
			threshold,draw=args.draw,norm_fac=1)        
	cv2.imwrite(args.file.replace(".png","_pred.png"), final_img)

if __name__ == '__main__':
	parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
	args = parser.parse_args()
	if args.file.endswith(".jpg") or args.file.endswith(".png"):
		main_image(args)
	else:
		main_video(args)