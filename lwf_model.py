import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.autograd import Variable
from Cifar100.resnet import resnet32
from Cifar100.Dataset.cifar100 import CIFAR100
import copy
import gc

from Cifar100 import utils

def auto_loss_rebalancing(n_known, n_classes, loss_type):
  alpha = n_known/n_classes 

  if loss_type == 'class':
    return 1-alpha
  return alpha

def get_rebalancing(rebalancing=None):
  if rebalancing is None:
    return lambda n_known, n_classes, loss_type: 1
  if rebalancing in ['auto', 'AUTO']:
    return auto_loss_rebalancing
  if callable(rebalancing):
    return rebalancing


# feature size: 2048 (currently)
# n_classes: 10 => 100
class LWF(nn.Module):
  def __init__(self, feature_size, n_classes, BATCH_SIZE, WEIGHT_DECAY, LR, GAMMA, NUM_EPOCHS, DEVICE,MILESTONES,MOMENTUM, \
      reverse_index = None, class_loss_criterion='bce', dist_loss_criterion='bce', loss_rebalancing='auto', lambda0=1):
    super(LWF, self).__init__()
    self.feature_extractor = resnet32()
    self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features,feature_size)
    self.bn = nn.BatchNorm1d(feature_size, momentum=MOMENTUM)
    self.ReLU = nn.ReLU()

    self.fc = nn.Linear(feature_size, n_classes, bias = False)

    self.n_classes = n_classes
    self.n_known = 0
    
    self.BATCH_SIZE = BATCH_SIZE
    self.WEIGHT_DECAY  = WEIGHT_DECAY
    self.LR = LR
    self.GAMMA = GAMMA # this allow LR to become 1/5 LR after MILESTONES epochs
    self.NUM_EPOCHS = NUM_EPOCHS
    self.DEVICE = DEVICE
    self.MILESTONES = MILESTONES # when the LR decreases, according to icarl
    self.MOMENTUM = MOMENTUM
    
    self.reverse_index=reverse_index
    
    self.class_loss, self.dist_loss = self.build_loss(class_loss_criterion, dist_loss_criterion, loss_rebalancing, lambda0=lambda0)

    self.optimizer, self.scheduler = utils.getOptimizerScheduler(self.LR, self.MOMENTUM, self.WEIGHT_DECAY, self.MILESTONES, self.GAMMA, self.parameters())

    gc.collect()
    
  def forward(self, x):
    x = self.feature_extractor(x)
    x = self.bn(x)
    x = self.ReLU(x)
    x = self.fc(x)

    return x
  
  # increment the number of classes considered by the net
  def increment_classes(self, n):
        gc.collect()
        """Add n classes in the final fc layer"""
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        self.fc = nn.Linear(in_features, out_features + n, bias = False)
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n

  def classify(self, images):
        """Classify images by softmax
        Args:
            x: input image batch
        Returns:
            preds: Tensor of size (batch_size,)
        """

        # for classification, lwf uses the network output values themselves
        gc.collect()
        _, preds = torch.max(torch.softmax(self.forward(images), dim=1), dim=1, keepdim=False)
        return preds    

  def update_representation(self, dataset, new_classes):
    #previous_model = copy.deepcopy(self)
    #previous_model.to(self.DEVICE)

    # 3 - increment classes
    #          (add output nodes)
    #          (update n_classes)
    gc.collect()

    self.increment_classes(len(new_classes))

    # define the loader for the augmented_dataset
    loader = DataLoader(dataset, batch_size=self.BATCH_SIZE,shuffle=True, num_workers=4, drop_last = True)
    
    self.cuda()

    net = self.feature_extractor
    net = net.to(self.DEVICE)

    optimizer = self.optimizer
    scheduler = self.scheduler

    criterion = utils.getLossCriterion()

    if self.n_known > 0:
        #old_net = copy.deepcopy(self.feature_extractor) #copy network before training
        old_net = copy.deepcopy(self) #test

    cudnn.benchmark # Calling this optimizes runtime
    for epoch in range(self.NUM_EPOCHS):
        print("NUM_EPOCHS: ",epoch,"/", self.NUM_EPOCHS)
        for indices, images, labels in loader:
            # Bring data over the device of choice
            images = images.to(self.DEVICE)
            #labels = self._one_hot_encode(labels, device=self.DEVICE)
            labels = labels.to(self.DEVICE)
            indices = indices.to(self.DEVICE)
            net.train()

            # PyTorch, by default, accumulates gradients after each backward pass
            # We need to manually set the gradients to zero before starting a new iteration
            optimizer.zero_grad() # Zero-ing the gradients

            # Forward pass to the network
            outputs = self.forward(images)

            # Loss = only classification on new classes
            loss = self.class_loss(outputs, labels, col_start=self.n_known)
            class_loss = loss.item() # Used for logging for debugging purposes

            # Distilation loss for old classes, class loss on new classes
            dist_loss = None
            if len(self.exemplar_sets) > 0:
              out_old = torch.sigmoid(old_net(images))
              dist_loss = self.dist_loss(outputs, out_old, col_end=self.n_known)
              loss += dist_loss

            loss.backward()
            optimizer.step()

        scheduler.step()
        print("LOSS: ", loss.item(), 'class loss', class_loss, 'dist loss', dist_loss.item() if dist_loss is not None else dist_loss)

    gc.collect()
    del net
    torch.no_grad()
    torch.cuda.empty_cache()



  def build_loss(self, class_loss_criterion, dist_loss_criterion, rebalancing=None, lambda0=1):
    class_loss_func = None
    dist_loss_func = None

    if class_loss_criterion in ['l2', 'L2']:
      class_loss_func = self.l2_class_loss
    elif class_loss_criterion in ['bce', 'BCE']:
      class_loss_func = self.bce_class_loss
    elif class_loss_criterion in ['ce', 'CE']:
      class_loss_func = self.ce_class_loss

    if dist_loss_criterion in ['l2', 'L2']:
      dist_loss_func = self.l2_dist_loss
    elif dist_loss_criterion in ['bce', 'BCE']:
      dist_loss_func = self.bce_dist_loss
    elif dist_loss_criterion in ['ce', 'CE']:
      dist_loss_func = self.ce_dist_loss

    rebalancing = get_rebalancing(rebalancing)
    
    def class_loss(outputs, labels, row_start=None, row_end=None, col_start=None, col_end=None):
      alpha = rebalancing(self.n_known, self.n_classes, 'class')
      return alpha*class_loss_func(outputs, labels, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end)
    
    def dist_loss(outputs, labels, row_start=None, row_end=None, col_start=None, col_end=None):
      alpha = rebalancing(self.n_known, self.n_classes, 'dist')
      return lambda0*alpha*dist_loss_func(outputs, labels, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end)
    
    return class_loss, dist_loss

  def bce_class_loss(self, outputs, labels, row_start=None, row_end=None, col_start=None, col_end=None):
    return self.bce_loss(outputs, labels, encode=True, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end)

  def bce_dist_loss(self, outputs, labels, row_start=None, row_end=None, col_start=None, col_end=None):
    return self.bce_loss(outputs, labels, encode=False, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end)

  def ce_class_loss(self, outputs, labels, row_start=None, row_end=None, col_start=None, col_end=None):
    return self.ce_loss(outputs, self.reverse_index.getNodes(labels), decode=False, row_start=row_start, row_end=row_end, col_start=None, col_end=col_end)
    
  def ce_dist_loss(self, outputs, labels, row_start=None, row_end=None, col_start=None, col_end=None):
    return self.ce_loss(outputs, labels, decode=True, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end)

  def l2_class_loss(self, outputs, labels, row_start=None, row_end=None, col_start=None, col_end=None):
    return self.bce_loss(outputs, labels, encode=True, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end)

  def l2_dist_loss(self, outputs, labels, row_start=None, row_end=None, col_start=None, col_end=None):
    return self.bce_loss(outputs, labels, encode=False, row_start=row_start, row_end=row_end, col_start=col_start, col_end=col_end)


  def bce_loss(self, outputs, labels, encode=False, row_start=None, row_end=None, col_start=None, col_end=None):
    criterion = nn.BCEWithLogitsLoss(reduction = 'mean')

    if encode:
      labels = utils._one_hot_encode(labels, self.n_classes, self.reverse_index, device=self.DEVICE)
      labels = labels.type_as(outputs)

    return criterion(outputs[row_start:row_end, col_start:col_end], labels[row_start:row_end, col_start:col_end])


  def ce_loss(self, outputs, labels, decode=False, row_start=None, row_end=None, col_start=None, col_end=None):
    criterion = nn.CrossEntropyLoss()

    if decode:
      labels = torch.argmax(labels, dim=1)
    
    return criterion(outputs[row_start:row_end, col_start:col_end], labels[row_start:row_end])


  def l2_loss(self, outputs, labels, encode=False, row_start=None, row_end=None, col_start=None, col_end=None):
    criterion = nn.MSELoss(reduction = 'mean')
    
    if encode:
      labels = utils._one_hot_encode(labels, self.n_classes, self.reverse_index, device=self.DEVICE)
      labels = labels.type_as(outputs)
    
    return criterion(outputs[row_start:row_end, col_start:col_end], labels[row_start:row_end, col_start:col_end])


