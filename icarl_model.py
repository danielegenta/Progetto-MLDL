"""
	This class implement the main model of iCaRL 
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
from torch.utils.data import DataLoader


from Cifar100.resnet import resnet34

# Hyper Parameters
# ...

# feature size: 2048
# n_classes: 10 => 100
class ICaRL(nn.Module):
  def __init__(self, feature_size, n_classes):
	   	# Network architecture
	    super(ICaRL, self).__init__()
	    self.feature_extractor = resnet34()

	    # this should maybe be changed
	    self.feature_extractor.fc =\
	        nn.Linear(self.feature_extractor.fc.in_features, feature_size)
	    self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
	    self.ReLU = nn.ReLU()
	    self.fc = nn.Linear(feature_size, n_classes)

	    self.n_classes = n_classes
	    self.n_known = 0
        self.batch_size = 128

	    # List containing exemplar_sets
	    # Each exemplar_set is a np.array of N images
	    # with shape (N, C, H, W)
	    self.exemplar_sets = []

	    # Learning method
	    """
	    self.cls_loss = nn.CrossEntropyLoss()
	                            self.dist_loss = nn.BCELoss()
	                            self.optimizer = optim.SGD(self.parameters(), lr=2.0,
	                                                       weight_decay=0.00001)
		"""

	    # Means of exemplars
	    self.compute_means = True
	    self.exemplar_means = []

  	
  def forward(self, x):
    x = self.feature_extractor(x)
    x = self.bn(x)
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

  # implementation of alg. 4 of icarl paper
  # iCaRL ConstructExemplarSet
  def construct_exemplar_set(self, tensors, m, transform):
    """Construct an exemplar set for image set
    Args:
        images: np.array containing images of a class
    """
    # Compute and cache features for each example
    features = []
    """
    for img in images:
                    x = Variable(transform(Image.fromarray(img)), volatile=True).cuda()
                    feature = self.feature_extractor(x.unsqueeze(0)).data.cpu().numpy()
                    feature = feature / np.linalg.norm(feature) # Normalize
                    features.append(feature[0])
    """
    loader = DataLoader(tensors,128,True,drop_last=False) #128 = batch size
    for images,labels in loader:
      feature = self.feature_extractor(images)  #(batchsize, 2048)

      # is this line important? it yields an error
      #feature = feature / np.linalg.norm(feature) # Normalize

      features.append(feature)

    features = np.array(features)
    class_mean = np.mean(features, axis=0)
    class_mean = class_mean / np.linalg.norm(class_mean) # Normalize

    exemplar_set = []
    exemplar_features = [] # list of Variables of shape (feature_size,)
    for k in range(1, m + 1):
        S = np.sum(exemplar_features, axis=0) # second addend, features in the exemplar set
        phi = features # first addend, all features
        mu = class_mean # current class mean
        mu_p = 1/k * (phi + S) 
        mu_p = mu_p / np.linalg.norm(mu_p) # normalize
        i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))) # finding the argming means finding the index of the image of the current class 
                                                                 # that'll be added to the exemplar set
        exemplar_set.append(images[i])
        exemplar_features.append(features[i])
        """
        print "Selected example", i
        print "|exemplar_mean - class_mean|:",
        print np.linalg.norm((np.mean(exemplar_features, axis=0) - class_mean))
        #features = np.delete(features, i, axis=0)
        """

    self.exemplar_sets.append(np.array(exemplar_set)) #update exemplar sets with the updated exemplars images

  def augment_dataset_with_exemplars(dataset):
    for y, P_y in enumerate(self.exemplar_sets):
        exemplar_images = P_y
        exemplar_labels = [y] * len(P_y) #i create a vector of labels [class class class ...] for each class in the exemplar set
        dataset.append(exemplar_images, exemplar_labels)
    return dataset

  # just a start to make the test work
  def update_representation(self, dataset, new_classes):
    # 1 - retrieve the classes from the dataset (which is the current train_subset)
    # 2 - retrieve the new classes
    # 1,2 are done in the main_icarl


    # 3 - increment classes
    #          (add output nodes)
    #          (update n_classes)
    self.increment_classes(len(new_classes))

    # 4 - combine current train_subset (dataset) with exemplars
    #     to form a new augmented train dataset
    augmented_dataset = augment_dataset_with_exemplars(dataset)

    # define the loader for the augmented_dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                               shuffle=True, num_workers=4)

    # 5 - store network outputs with pre-update parameters = ???

    # 6 - run network training, with loss function

    # todo ...


  # implementation of alg. 5 of icarl paper
  # iCaRL ReduceExemplarSet
  def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
          # i keep only the first m exemplar images
          # where m is the UPDATED K/number_classes_seen
          # the number of images per each exemplar set (class) progressively decreases
          self.exemplar_sets[y] = P_y[:m] 