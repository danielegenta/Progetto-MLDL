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

# Hyper Parameters
# ...

# feature size: ???
# n_classes: 10 => 100
class LWF(nn.Module):
  def __init__(self, feature_size, n_classes, BATCH_SIZE, WEIGHT_DECAY, LR, GAMMA, NUM_EPOCHS, DEVICE,MILESTONES):
    super(LWF, self).__init__()
    self.feature_extractor = resnet32()

    self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features,feature_size)
    #self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
    self.ReLU = nn.ReLU()
    self.fc = nn.Linear(feature_size, n_classes, bias = False)

    self.n_classes = n_classes
    self.n_known = 0

    self.cls_loss = nn.BCEWithLogitsLoss(reduction = 'mean')
    self.dist_loss = nn.BCEWithLogitsLoss(reduction = 'mean')
    
    self.BATCH_SIZE = BATCH_SIZE
    self.WEIGHT_DECAY  = WEIGHT_DECAY
    self.LR = LR
    self.GAMMA = GAMMA # this allow LR to become 1/5 LR after MILESTONES epochs
    self.NUM_EPOCHS = NUM_EPOCHS
    self.DEVICE = DEVICE

    self.MILESTONES = MILESTONES # when the LR decreases, according to icarl
    self.optimizer = optim.SGD(self.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)
    self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=MILESTONES, gamma=self.GAMMA)

    
  def forward(self, x):
    x = self.feature_extractor(x)
    #x = self.bn(x)
    x = self.ReLU(x)
    x = self.fc(x)

    return x
  
  # increment the number of classes considered by the net
  def increment_classes(self, n):
        """Add n classes in the final fc layer"""
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        self.fc = nn.Linear(in_features, out_features+n, bias=False)
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
        _, preds = torch.max(torch.softmax(self.forward(images), dim=1), dim=1, keepdim=False)
        return preds    


  def _one_hot_encode(self, labels, dtype=None, device=None):
    enconded = torch.zeros(self.n_classes, len(labels), dtype=dtype, device=device)
    for i, l in enumerate(labels):
      enconded[i, l] = 1
    return enconded

  def update_representation(self, dataset, new_classes):
    #previous_model = copy.deepcopy(self)
    #previous_model.to(self.DEVICE)
    self.to(self.DEVICE)

    # 3 - increment classes
    #          (add output nodes)
    #          (update n_classes)
    self.increment_classes(len(new_classes))

    # define the loader for the augmented_dataset
    loader = DataLoader(dataset, batch_size=self.BATCH_SIZE,shuffle=True, num_workers=4, drop_last = True)

    # 5 - store network outputs with pre-update parameters => q
    q = torch.zeros(len(dataset), self.n_classes)
    for indices, images, labels in loader:
        images = Variable(images).to(self.DEVICE)
        indices = indices.to(self.DEVICE)
        g = nn.functional.sigmoid(self.forward(images))
        q_i = g.data
        q[indices] = q_i
    q = Variable(q).to(self.DEVICE)

    optimizer = self.optimizer


    for epoch in range(NUM_EPOCHS):
        for indices, images, labels in loader:
            # Bring data over the device of choice
            images = Variable(images).to(self.DEVICE)
            #labels = self._one_hot_encode(labels, device=self.DEVICE)
            labels = Variable(labels_onehot).to(self.DEVICE)
            indices = indices.to(self.DEVICE)

            # PyTorch, by default, accumulates gradients after each backward pass
            # We need to manually set the gradients to zero before starting a new iteration
            optimizer.zero_grad() # Zero-ing the gradients

            # Forward pass to the network
            outputs = self.forward(images)

            # Classification loss for new classes
            labels_onehot = nn.functional.one_hot(labels,n_classes)
            labels_onehot = labels_onehot.type_as(outputs)
            loss = sum(self.cls_loss(g[:,y], labels_onehot[:,y]) for y in range(self.n_known, self.n_classes))

            # Distillation loss for old classes
            # Distilation loss for old classes
            if self.n_known > 0:
                g = F.sigmoid(g)
                q_i = q[indices]
                # to check!
                for y in range(0,len(self.exemplar_sets)):
                    dist_loss += self.dist_loss(g[:,y],q_i[:,y])
                loss += dist_loss

            loss.backward()
            optimizer.step()
