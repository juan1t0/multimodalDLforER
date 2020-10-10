import torch
import torch.nn as nn

import numpy as np

from models.face_net import Model as FaceNN
from models.skeleton_net import Model as PoseNN
from models.context_net import resnet18

'''
  Revision general / comentar
'''

class LinearFusion(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(LinearFusion, self).__init__()
    self.lnr_1a = nn.Linear(in_channels, 32)
    self.lnr_1b = nn.Linear(in_channels, 32)
    self.lnr_2 = nn.Sequential( nn.Linear(64,128),
                                nn.Linear(128, 128),
                                nn.Linear(128, out_channels))
  
  def forward(self, input_A, input_B):
    inputA = self.lnr_1a(input_A)
    inputB = self.lnr_1b(input_B)
    input = torch.cat((inputA, inputB))
    output = self.lnr_2(input)
    return output

class Model(nn.Module):
  def __init__(self, n_classes, inner_models_config, beta=1.0, weights=None):
    super(Model, self).__init__()
    self.NClasses = n_classes
    self.MF_Beta = beta
    if weights is None:
      self.WS_weights = self.gen_weights()
    else:
      self.WS_weights = weights

    self.FLM_model = self.gen_model(inner_models_config['FLM_model'])
    self.SKL_model = self.gen_model(inner_models_config['SKL_model'])
    self.CTX_model = self.gen_model(inner_models_config['CTX_model'])
    self.FusionModule = self.gen_fusion_module(inner_models_config['Fusion_model'])

  def gen_weights(self):
    w = [0.3, 0.5, 0.5]
    return w

  def gen_model(self, model_config):
    if model_config['name'] == 'FaceNN':
      Model = FaceNN(inchannels=model_config['in_channels'],
                     outchannels=model_config['out_channels'])
    elif model_config['name'] == 'PoseNN':
      Model = PoseNN(num_class=model_config['num_class'],
                     num_point=model_config['num_point'],
                     num_person=model_config['num_person'],
                     in_channels=model_config['in_channels'])
    elif model_config['name'] == 'ABNN':
      Model = resnet18(pretrained=model_config['pretrain'],
                       num_classes=model_config['num_classes'])
    else:
      raise Exception('Model have not supported yet')
    return Model
  
  def gen_fusion_module(self, model_config):
    if model_config['type'] == 'Linear':
      LinearModule = LinearFusion(in_channels=model_config['in_channels'],
                                  out_channels=model_config['out_channels'])
      return LinearModule
    else:
      raise Exception('Module have not supported yet')

  def WeightedSum(self, modalities):
    fused = torch.zeros(modalities[0].shape)

    for i, P in enumerate(modalities):
      fused += P * self.WS_weights[i]

    return fused
  
  def MultiplicativeFusion(self, modalities):
    fused = torch.zeros(modalities[0].shape)
    n = float(len(modalities))
    for i, P in enumerate(modalities):
      P1 = P **(self.MF_Beta/n- 1.)
      P2 = torch.log(P)
      fused = fused + (P1*P2)

    return fused * (-1)

  def forward(self, input_flm, input_sklj, input_sklb, input_ctx):
    out_flm = self.FLM_model.forward(input_flm)
    out_skl = self.SKL_model.forward(input_sklj, input_sklb)
    out_ctx = self.CTX_model.forward(input_ctx)

    weighsum = self.WeightedSum([out_flm, out_skl, out_ctx])
    multfusi = self.MultiplicativeFusion([out_flm, out_skl, out_ctx])

    output = self.FusionModule.forward(weighsum, multfusi)

    return output, (out_flm, out_skl, out_ctx, (weighsum, multfusi))