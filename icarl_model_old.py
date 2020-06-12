"""
    This class implements the main model of iCaRL 
    and all the methods regarding the exemplars
    from delivery: iCaRL is made up of 2 components
    - feature extractor (a convolutional NN) => resnet32 optimized on cifar100
    - classifier => a FC layer OR a non-parametric classifier (NME)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn
from torch.autograd import Variable
import copy
import gc #extensive use in order to manage memory issues
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToPILImage 

from Cifar100 import utils
from Cifar100.resnet import resnet32
from Cifar100.Dataset.cifar100 import CIFAR100
import random
import pandas as pd

# new classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# feature_size: 2048, why?
# n_classes: 10 => 100
class ICaRL(nn.Module):
  def __init__(self, feature_size, n_classes, BATCH_SIZE, WEIGHT_DECAY, LR, GAMMA, NUM_EPOCHS, DEVICE,MILESTONES,MOMENTUM,K, herding, reverse_index = None):
    super(ICaRL, self).__init__()
    self.net = resnet32()
    self.net.fc = nn.Linear(self.net.fc.in_features, n_classes)

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
    
    # List containing exemplar_sets
    # Each exemplar_set is a np.array of N images
    self.exemplar_sets = []
    self.exemplar_sets_indices = []

    
    # for the classification/distillation loss we have two alternatives
    # 1- BCE loss with Logits (reduction could be mean or sum)
    # 2- BCE loss + sigmoid
    # actually we use just one loss as explained on the forum
    

    # Means of exemplars (cntroids)
    self.compute_means = True
    self.exemplar_means = []

    self.herding = herding # random choice of exemplars or icarl exemplars strategy?

    # this is used as explained in the forum to compute the exemplar mean in a more accurate way
    # populated during construct exemplar set and used in the classify step
    self.data_from_classes = []
    self.means_from_classes = []

    # Knn, svc classification
    self.model = None
  
  # increment the number of classes considered by the net
  # incremental learning approach, 0,10..100
  def increment_classes(self, n):
        gc.collect()

        in_features = self.net.fc.in_features
        out_features = self.net.fc.out_features
        weights = self.net.fc.weight.data
        bias = self.net.fc.bias.data

        self.net.fc = nn.Linear(in_features, out_features + n) #add 10 classes to the fc last layer
        self.net.fc.weight.data[:out_features] = weights
        self.net.fc.bias.data[:out_features] = bias
        self.n_classes += n #icrement #classes considered

  # computes the mean of each exemplar set
  def computeMeans(self):
    torch.no_grad()  
    torch.cuda.empty_cache()

    feature_extractor = self.feature_extractor.to(self.DEVICE)
    feature_extractor.train(False)

    # new mean mgmt
    tensors_mean = []
    with torch.no_grad():
      for tensor_set in self.data_from_classes:
        features = []
        for tensor, _ in tensor_set:
          
          tensor = tensor.to(self.DEVICE)
          feature = feature_extractor(tensor)

          feature.data = feature.data / feature.data.norm() # Normalize
          features.append(feature)

          # cleaning 
          torch.no_grad()
          torch.cuda.empty_cache()

        features = torch.stack(features) #(num_exemplars,num_features)
        mean_tensor = features.mean(0) 
        mean_tensor.data = mean_tensor.data / mean_tensor.data.norm() # Re-normalize
        mean_tensor = mean_tensor.to('cpu')
        tensors_mean.append(mean_tensor)

    self.exemplar_means = tensors_mean  # nb the mean is computed over all the imgs

    # cleaning
    torch.no_grad()  
    torch.cuda.empty_cache()

  # train procedure common for KNN and SVC classifier (save a lot of training time)
  def modelTrain(self, method, K_nn = None):
    torch.no_grad()
    torch.cuda.empty_cache()

    feature_extractor = self.feature_extractor.to(self.DEVICE)
    feature_extractor.train(False)

    # -- train a SVC classifier
    X_train, y_train = [], []

    for exemplar_set in self.exemplar_sets:
          for exemplar, label in  exemplar_set:
            exemplar = exemplar.to(self.DEVICE)
            feature = feature_extractor(exemplar)
            feature = feature.squeeze()
            feature.data = feature.data / feature.data.norm() # Normalize
            X_train.append(feature.cpu().detach().numpy())
            y_train.append(label)
    
    if method == 'KNN':
      model = KNeighborsClassifier(n_neighbors = K_nn)
    elif method == 'SVC':
      model = LinearSVC()
    self.model = model.fit(X_train, y_train)

  # common classify function
  def KNN_SVC_classify(self, images):
    torch.no_grad()
    torch.cuda.empty_cache()

    # --- prediction
    X_pred = []
    images = images.to(self.DEVICE)
    feature_extractor = self.feature_extractor.to(self.DEVICE)
    feature_extractor.train(False)

    features = feature_extractor(images)
    for feature in features:
      feature = feature.squeeze()
      feature.data = feature.data / feature.data.norm() # Normalize
      X_pred.append(feature.cpu().detach().numpy())
    
    preds = self.model.predict(X_pred)
    # --- end prediction
    return torch.tensor(preds)
  
  # classification via fc layer (similar to lwf approach)
  def FCC_classify(self, images):
    _, preds = torch.max(torch.softmax(self.net(images), dim=1), dim=1, keepdim=False)
    return preds
  # NME classification from iCaRL paper
  def classify(self, batch_imgs):
      """Classify images by nearest-mean-of-exemplars
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

      # update exemplar_means with the mean
      # of all the train data for a given class

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
  def construct_exemplar_set(self, tensors, m, label):
    """
      Args:
          tensors: train_subset containing a single label
          m: number of exemplars allowed/exemplar set (class)
          label: considered class
    """
    torch.no_grad()
    torch.cuda.empty_cache()
    gc.collect()

    exemplar_set_indices = set()
    exemplar_list_indices = []
    exemplar_set = []
    if self.herding:

      feature_extractor = self.feature_extractor.to(self.DEVICE)
      feature_extractor.train(False)

      # Compute and cache features for each example
      features = []

      loader = DataLoader(tensors,batch_size=self.BATCH_SIZE,shuffle=True,drop_last=False,num_workers = 4)

      with torch.no_grad():
        for _, images, labels in loader:
          images = images.to(self.DEVICE)
          labels = labels.to(self.DEVICE)
          feature = feature_extractor(images) 

          feature = feature / np.linalg.norm(feature.cpu()) # Normalize
          
          features.append(feature)

      features_s = torch.cat(features)
      
      class_mean = features_s.mean(0)
      class_mean = class_mean / np.linalg.norm(class_mean.cpu()) # Normalize
      class_mean = torch.stack([class_mean]*features_s.size()[0])

      summon = torch.zeros(1,features_s.size()[1]).to(self.DEVICE) #(1,num_features)
      for k in range(1, (m + 1)):
          S = torch.cat([summon]*features_s.size()[0]) # second addend, features in the exemplar set
          results = pd.DataFrame((class_mean-(1/k)*(features_s + S)).pow(2).sum(1).cpu(), columns=['result']).sort_values('result')
          results['index'] = results.index
          results = results.to_numpy()

          # select argmin not included in exemplar_set_indices
          for i in range(results.shape[0]):
            index = results[i, 1]
            exemplar_k_index = tensors[index][0]
            if exemplar_k_index not in exemplar_set_indices:
              exemplar_k = tensors[index][1].unsqueeze(dim = 0) # take the image from the tuple (index, img, label)
              exemplar_set.append((exemplar_k, label))
              exemplar_k_index = tensors[index][0] # index of the img on the real dataset
              
              exemplar_list_indices.append(exemplar_k_index)
              exemplar_set_indices.add(exemplar_k_index)
              break

          # features of the exemplar k
          phi = feature_extractor(exemplar_k.to(self.DEVICE)) #feature_extractor(exemplar_k.to(self.DEVICE))
          summon += phi # update sum of features
    else:
      tensors_size = len(tensors)
      unique_random_indexes = random.sample(range(0, tensors_size), m) # random sample without replacement k exemplars
      i = 0
      for k in range(1, (m + 1)):
        index = unique_random_indexes[i]
        exemplar_k = tensors[index][1].unsqueeze(dim = 0)
        exemplar_k_index = tensors[index][0]
        exemplar_set.append((exemplar_k, label))
        exemplar_set_indices.add(exemplar_k_index)
        i = i + 1

    # --- new ---
    tensor_set = []
    for i in range(0, len(tensors)):
      t = tensors[i][1].unsqueeze(dim = 0)
      tensor_set.append((t, label))
    
    self.exemplar_sets.append(exemplar_set) #update exemplar sets with the updated exemplars images
    self.exemplar_sets_indices.append(exemplar_list_indices)

    # this is used to compute more accurately the means of the exemplar (see also computeMeans and classify)
    self.data_from_classes.append(tensor_set)

    # cleaning
    torch.cuda.empty_cache()

  # build a exemplar dataset as a subset of the train dataset
  def build_exemplars_dataset(self, train_dataset): #complete train dataset
    all_exemplars_indices = []
    for exemplar_set_indices in self.exemplar_sets_indices:
        all_exemplars_indices.extend(exemplar_set_indices)

    exemplars_dataset = Subset(train_dataset, all_exemplars_indices)
    return exemplars_dataset

  def update_representation(self, dataset, train_dataset_big, new_classes):
    # 1 - retrieve the classes from the dataset (which is the current train_subset)
    # 2 - retrieve the new classes
    # 1,2 are done in the main_icarl
    #gc.collect()

    # 3 - increment classes
    #          (add output nodes)
    #          (update n_classes)
    # 5        store network outputs with pre-update parameters
    self.increment_classes(len(new_classes))

    # 4 - combine current train_subset (dataset) with exemplars
    #     to form a new augmented train dataset
    # join the datasets
    exemplars_dataset = self.build_exemplars_dataset(train_dataset_big)
    #
    if len(exemplars_dataset) > 0:
      augmented_dataset = ConcatDataset(dataset, exemplars_dataset)
      #augmented_dataset = utils.joinSubsets(train_dataset_big, [dataset, exemplars_dataset])
    else: 
      augmented_dataset = dataset # first iteration

    # 6 - run network training, with loss function

    net = self.net

    optimizer = optim.SGD(net.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY, momentum=self.MOMENTUM)
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
        for _, images, labels in loader:
            # Bring data over the device of choice
            images = images.to(self.DEVICE)
            labels = labels.to(self.DEVICE)
            net.train()

            # PyTorch, by default, accumulates gradients after each backward pass
            # We need to manually set the gradients to zero before starting a new iteration
            optimizer.zero_grad() # Zero-ing the gradients

            # Forward pass to the network
            outputs = net(images)

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
        print("LOSS: ", loss.item())

    self.net = copy.deepcopy(net)
    self.feature_extractor = copy.deepcopy(net)
    self.feature_extractor.fc = nn.Sequential()

    #cleaning
    del net
    torch.cuda.empty_cache()


  # implementation of alg. 5 of icarl paper
  # iCaRL ReduceExemplarSet
  def reduce_exemplar_sets(self, m):
  	    # i keep only the first m exemplar images
        # where m is the UPDATED K/number_classes_seen
        # the number of images per each exemplar set (class) progressively decreases
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m] 
        for x, P_x in enumerate(self.exemplar_sets_indices):
            self.exemplar_sets_indices[x] = P_x[:m] 


# ---------- 
from torch.utils.data import Dataset
"""
  Merge two different datasets (train and exemplars in our case)
  format:
  train
  --------
  exemplars
  train leans on cifar100
  exemplars is managed here (exemplar_transform is performed) => changed
"""
class ConcatDataset(Dataset):
    
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.l1 = len(dataset1)
        self.l2 = len(dataset2)

    def __getitem__(self,index):
        if index < self.l1:
            _, image,label = self.dataset1[index] #here it leans on cifar100 get item
            return _, image,label
        else:
            _, image, label = self.dataset2[index - self.l1]
            return _, image,label

    def __len__(self):
        return (self.l1 + self.l2)
#------------