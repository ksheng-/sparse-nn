# sparse-neural-network
"Learning Sparse Neural Networks through L_0 Regularization" by Christos Louizos, Max Welling, Diederik P. Kingma\
https://arxiv.org/pdf/1712.01312.pdf\
https://github.com/AMLab-Amsterdam/L0_regularization\

## Reproducing part of Table 1: using L0 regularization to prune LeNet-5-Caffe
Pruning the original 20-50-800-500 architecture to about 9-18-65-25 with 99% accuracy.
The important part is the level of shrinkage achieved in the computationally expensive
fully connected layers.

## Results:
```
Deterministic pruned architecture after 110001 global steps: 14-19-36-21
Test accuracy: 0.9872999787330627
Test loss: 0.30946531891822815
```

## Example of pruning at train time (one arbitrary step):
```
112599/1100000 [15:38<2:17:10, 119.97it/s, epoch=204, neurons=[14.0, 19.0, 35.0, 21.0], t_acc=0.999, t_loss=ø0.223, v_acc=0.986]
```
