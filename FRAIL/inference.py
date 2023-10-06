from PIL import Image
from FRAIL.models.ghostnet import GhostNet, ghost_net
from torchvision import transforms #to pre-process the image from the captured frame.

image_path = ''
image = Image.open(image_path)
transforms = transforms.Compose()
input_data = transforms(image)

model = ghost_net()
model.eval()
print(model(input_data))

