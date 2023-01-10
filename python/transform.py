from PIL import Image
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
