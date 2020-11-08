import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision.models import vgg19

cfg = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class ShortVGG(nn.Module):
	def __init__(self, vgg_name, numclasses):
		super(ShortVGG, self).__init__()
		self.features = self._make_layers(cfg[vgg_name])
		self.classifier = nn.Linear(512, numclasses)

	def forward(self, x):
		out = self.features(x)
		out = out.view(out.size(0), -1)
		out = F.dropout(out, p=0.5, training=self.training)
		out = self.classifier(out)
		return out

	def _make_layers(self, cfg):
		layers = []
		in_channels = 3
		for x in cfg:
			if x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
									 nn.BatchNorm2d(x),
									 nn.ReLU(inplace=True)]
				in_channels = x
		
		layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
		return nn.Sequential(*layers)

def getTorchVGG(numclasses, pretrained=False):
	model = vgg19(pretrained)
	model.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
																	 nn.ReLU(True),
																	 nn.Dropout(),
																	 nn.Linear(4096, 1024),
																	 nn.ReLU(True),
																	 nn.Dropout(),
																	 nn.Linear(1024, numclasses))
	
	return model

class SimpleModel (nn.Module):
	def __init__(self, inchanels=144, outchanels=26) :
		super(SimpleModel, self).__init__()
		self.inchanels = inchanels
		self.outchanels = outchanels

		self.conv_layers = nn.Sequential(
				nn.Conv1d(inchanels, 256, kernel_size=1),
				nn.BatchNorm1d(256),
				nn.ReLU(),

				nn.Conv1d(256, 512, kernel_size=1),
				nn.BatchNorm1d(512),
				nn.ReLU(),

				nn.Conv1d(512, 1024, kernel_size=1),
				nn.BatchNorm1d(1024),
				nn.ReLU(),

				nn.MaxPool1d(kernel_size=1, stride=1)
		)
		self.lner_layers = nn.Sequential(
				nn.Linear(1024, 1024),
				nn.Linear(1024, 512),
				nn.Linear(512, 256),
				nn.Linear(256, outchanels)
		)
	def forward(self, x):
		x = self.conv_layers(x)
		x = x.view(x.size(0), -1)
		x = self.lner_layers(x)
		return x