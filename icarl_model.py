"""
    This class implements the main model of iCaRL 
    and all the methods regarding the exemplars

    from delivery: iCaRL is made up of 2 components
    - feature extractor (a convolutional NN) => resnet32 optimized on cifar100
    - classifier => a FC layer OR a non-parametric classifier (NME)

    main ref: https://github.com/donlee90/icarl
"""


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
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToPILImage 

from Cifar100 import utils


# Hyper Parameters
# ...

# feature size: 2048
# n_classes: 10 => 100
class ICaRL(nn.Module):
  def __init__(self, feature_size, n_classes, BATCH_SIZE, WEIGHT_DECAY, LR, GAMMA, NUM_EPOCHS, DEVICE,MILESTONES,MOMENTUM,K, transform, reverse_index = None):
    super(ICaRL, self).__init__()
    self.net = resnet32()
    self.net.fc = nn.Linear(feature_size, n_classes)

    self.feature_extractor = resnet32()
    self.feature_extractor.fc = nn.Sequential()

    self.n_classes = n_classes
    self.n_known = 0

    # Hyper-parameters from iCaRL
    self.BATCH_SIZE = BATCH_SIZE
    self.WEIGHT_DECAY  = WEIGHT_DECAY
    self.LR = LR
    self.GAMMA = GAMMA # this allow LR to become 1/5 LR after MILESTONES epochs
    self.NUM_EPOCHS = NUM_EPOCHS
    self.DEVICE = DEVICE
    self.MILESTONES = MILESTONES # when the LR decreases, according to icarl
    self.MOMENTUM = MOMENTUM
    self.K = K
    
    self.reverse_index=reverse_index

    self.optimizer, self.scheduler = utils.getOptimizerScheduler(self.LR, self.MOMENTUM, self.WEIGHT_DECAY, self.MILESTONES, self.GAMMA, self.parameters())

    gc.collect()
    
    self.transform = transform

    # List containing exemplar_sets
    # Each exemplar_set is a np.array of N images
    # with shape (N, C, H, W)
    self.exemplar_sets = []

    # Learning method
    
    # for the classification loss we have two alternatives
    # 1- BCE loss with Logits (reduction could be mean or sum)
    # 2- BCE loss + sigmoid
    """self.cls_loss = nn.BCEWithLogitsLoss(reduction = 'mean')
                self.dist_loss = nn.BCEWithLogitsLoss(reduction = 'mean')"""
    
    

    # Means of exemplars
    self.compute_means = True
    self.exemplar_means = []
  
  # increment the number of classes considered by the net
  def increment_classes(self, n):
        gc.collect()

        """Add n classes in the final fc layer"""
        in_features = self.net.fc.in_features
        out_features = self.net.fc.out_features
        weight = self.net.fc.weight.data

        self.net.fc = nn.Linear(in_features, out_features + n, bias = False)
        self.feature_extractor.weight.data[:out_features] = weight
        self.n_classes += n

  # computes the means of each exemplar set
  def computeMeans(self):
    torch.no_grad()  
    torch.cuda.empty_cache()

    exemplar_means = []
    feature_extractor = self.feature_extractor.to(self.DEVICE)
    feature_extractor.train(False)

    with torch.no_grad():
      for exemplar_set in self.exemplar_sets:
        features=[]
        for exemplar in exemplar_set:
          exemplar = exemplar.to(self.DEVICE)
          feature = feature_extractor(exemplar)
          features.append(feature)

          # cleaning 
          torch.no_grad()
          torch.cuda.empty_cache()

        features = torch.stack(features) # (num_exemplars,num_features)
        mean_exemplar = features.mean(0) 
        mean_exemplar.data = mean_exemplar.data / mean_exemplar.data.norm() # Normalize
        mean_exemplar = mean_exemplar.to('cpu')
        exemplar_means.append(mean_exemplar)

        # cleaning
        torch.no_grad()  
        torch.cuda.empty_cache()

    self.exemplar_means = exemplar_means


  def classify(self, batch_imgs):
      """Classify images by neares-means-of-exemplars
      Args:
          batch_imgs: input image batch
      Returns:
          preds: Tensor of size (batch_size,)
      """
      torch.no_grad()
      torch.cuda.empty_cache()

      batch_imgs_size = batch_imgs.size(0)
      feature_extractor = self.feature_extractor.to(self.DEVICE)
      feature_extractor.train(False)

      means_exemplars = torch.cat(self.exemplar_means, dim=0)
      means_exemplars = torch.stack([means_exemplars] * batch_imgs_size)
      means_exemplars = means_exemplars.transpose(1, 2) 

      feature = feature_extractor(batch_imgs) 
      aus_normalized_features = []
      for el in feature: # Normalize
          el.data = el.data / el.data.norm()
          aus_normalized_features.append(el)

      feature = torch.stack(aus_normalized_features,dim=0)

      feature = feature.unsqueeze(2) 
      feature = feature.expand_as(means_exemplars) 

      means_exemplars = means_exemplars.to(self.DEVICE)
      # Nearest prototype
      preds = torch.argmin((feature - means_exemplars).pow(2).sum(1),dim=1)

      # cleaning
      torch.no_grad()
      torch.cuda.empty_cache()
      gc.collect()

      return preds

  # implementation of alg. 4 of icarl paper
  # iCaRL ConstructExemplarSet
  def construct_exemplar_set(self, tensors, m, transform,label):
    torch.no_grad()
    torch.cuda.empty_cache()
    gc.collect()

    feature_extractor = self.feature_extractor.to(self.DEVICE)
    feature_extractor.train(False)

    """Construct an exemplar set for image set
    Args:
        images: np.array containing images of a class
    """
    # Compute and cache features for each example
    features = []

    loader = DataLoader(tensors,batch_size=self.BATCH_SIZE,shuffle=True,drop_last=False,num_workers = 4)

    for _, images, labels in loader:
      images = images.to(self.DEVICE)
      labels = labels.to(self.DEVICE)
      feature = feature_extractor(images) 

      # is this line important? it yields an error
      #feature = feature / np.linalg.norm(feature) # Normalize

      features.append(feature)

    features_s = torch.cat(features)
    class_mean = features_s.mean(0)
    class_mean = torch.stack([class_mean]*features_s.size()[0])
    torch.cuda.empty_cache()

    exemplar_set = []
    exemplar_features = [] # list of Variables of shape (feature_size,)
    summon = torch.zeros(1,features_s.size()[1]).to(self.DEVICE) #(1,num_features)
    for k in range(1, (m + 1)):
        S = torch.cat([summon]*features_s.size()[0]) # second addend, features in the exemplar set
        i = torch.argmin((class_mean-(1/k)*(features_s + S)).pow(2).sum(1),dim=0)
        exemplar_k = tensors[i.item()][1].unsqueeze(dim = 0) # take the image from the tuple (index, img, label)
        
        exemplar_set.append((exemplar_k, label))

        # test features of the exemplar
        phi = feature_extractor(exemplar_k.to(self.DEVICE)) #feature_extractor(exemplar_k.to(self.DEVICE))
        summon += phi # update sum of features
        del exemplar_k 

    # cleaning
    torch.cuda.empty_cache()
    self.exemplar_sets.append(exemplar_set) #update exemplar sets with the updated exemplars images


  def augment_dataset_with_exemplars(self, dataset):
    transformToImg = transforms.ToPILImage()
    index = 0
    aus_dataset = []
    for exemplar_set in self.exemplar_sets: #for each class and exemplar set for that class
        for exemplar, label in exemplar_set:
            exemplar = exemplar.squeeze()
            #print(exemplar.size())
            img = ToPILImage()(exemplar)
            #exemplar = Image.fromarray(np.array(exemplar)) # Return a PIL image
            aus_dataset.append((img, label)) # nb i do not append the label yet a simple index, 0 is just a placeholder
        index += 1

    return aus_dataset 

  def _one_hot_encode(self, labels, dtype=None, device=None):
    enconded = torch.zeros(self.n_classes, len(labels), dtype=dtype, device=device)
    for i, l in enumerate(labels):
      enconded[i, l] = 1
    return enconded

  def update_representation(self, dataset, new_classes):
    #print(new_classes)
    # 1 - retrieve the classes from the dataset (which is the current train_subset)
    # 2 - retrieve the new classes
    # 1,2 are done in the main_icarl
    #gc.collect()

    # 3 - increment classes
    #          (add output nodes)
    #          (update n_classes)

    #self.increment_classes(len(new_classes))
    self.n_classes += len(new_classes)

    # 4 - combine current train_subset (dataset) with exemplars
    #     to form a new augmented train dataset
    exemplars_dataset = self.augment_dataset_with_exemplars(dataset)
    augmented_dataset = ConcatDataset(dataset, exemplars_dataset, self.transform)

    # join the datasets

    #self.cuda()
    # 5 - store network outputs with pre-update parameters => q
    # 6 - run network training, with loss function

    net = self.net

    optimizer = optim.SGD(net.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY, momentum=self.MOMENTUM)

    # Scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.MILESTONES, gamma=self.GAMMA, last_epoch=-1)

    criterion = utils.getLossCriterion()

    cudnn.benchmark # Calling this optimizes runtime
    net = net.to(self.DEVICE)

    # define the loader for the augmented_dataset
    loader = DataLoader(augmented_dataset, batch_size=self.BATCH_SIZE,shuffle=True, num_workers=4, drop_last = True)

    if len(self.exemplar_sets) > 0:
      old_net = copy.deepcopy(net) 
    for epoch in range(self.NUM_EPOCHS):
        print("NUM_EPOCHS: ",epoch,"/", self.NUM_EPOCHS)
        for images, labels in loader:
            # Bring data over the device of choice
            images = images.to(self.DEVICE)
            labels = labels.to(self.DEVICE)
            net.train()

            # PyTorch, by default, accumulates gradients after each backward pass
            # We need to manually set the gradients to zero before starting a new iteration
            optimizer.zero_grad() # Zero-ing the gradients

            # Forward pass to the network
            outputs = net(images)
            if self.n_known > 0:
              print(outputs.size())

            labels_one_hot = utils._one_hot_encode(labels, self.n_classes, self.reverse_index, device=self.DEVICE)
            labels_one_hot = labels_one_hot.type_as(outputs)

            # Loss = only classification on new classes
            if len(self.exemplar_sets) == 0:
                loss = criterion(outputs, labels_one_hot)
            # Distilation loss for old classes, class loss on new classes
            if len(self.exemplar_sets) > 0:

               labels_one_hot = labels_one_hot.type_as(outputs)[:,len(self.exemplar_sets):]
               out_old = Variable(torch.sigmoid(old_net(images))[:,:len(self.exemplar_sets)],requires_grad = False)

               #[outputold, onehot_new]
               target = torch.cat((out_old, labels_one_hot),dim=1)
               loss = criterion(outputs,target)

            loss.backward()
            optimizer.step()

        scheduler.step()
        print("LOSS: ",loss)

    self.net = copy.deepcopy(net)
    self.feature_extractor = copy.deepcopy(net)
    self.feature_extractor.fc = nn.Sequential()
    del net
    #gc.collect()
    #
    #torch.no_grad()
    #torch.cuda.empty_cache()


  # implementation of alg. 5 of icarl paper
  # iCaRL ReduceExemplarSet
  def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            # i keep only the first m exemplar images
            # where m is the UPDATED K/number_classes_seen
            # the number of images per each exemplar set (class) progressively decreases
            self.exemplar_sets[y] = P_y[:m] 


# ----------
from torch.utils.data import Dataset
"""
  Merge two different datasets (train and exemplars in our case)
"""
class ConcatDataset(Dataset):
    
    def __init__(self, dataset1, dataset2, transform):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.transform = transform
        self.l1 = len(dataset1)
        self.l2 = len(dataset2)

    def __getitem__(self,index):
        if index < self.l1:
            _, image,label = self.dataset1[index]
            return image,label
        else:
            image, label = self.dataset2[index - self.l1]
            image = self.transform(image)
            return image,label

    def __len__(self):
        return (self.l1 + self.l2)

#------------