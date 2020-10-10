# Incremental learning in image classification: an ablation study

## Incremental learning is a learning paradigm in which a deep architecture is required to continually learn from a stream of data.

In our work, we implemented several state-of-the-art algorithms for incremental learning, such as: Fine Tuning, Learning Without Forgetting and [iCaRL](https://arxiv.org/abs/1611.07725). 
Then, we made several updates to the origial iCaRL algorithm: in particular, we experiment with different combinations of distillation and classification losses and introduce new classifiers into the framework.
Furthermore, we propose some extensions of the origial iCaRL algorithm and we verify their effectiveness. We perform our tests on [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), as used in the original iCaRL paper.

Final resulting paper is available [here](https://github.com/danielegenta/Progetto-MLDL/blob/master/Report/Genta_Massimino_Paesante.pdf)

## Frameworks used

-PyTorch
-scikit-learn
