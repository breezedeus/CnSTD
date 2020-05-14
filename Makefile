NUM_GPU = 2

train:
	python train.py data/icdar2015/train ~/.mxnet/models/resnet50_v1b-0ecdba34.params $(NUM_GPU)

.PHONY: train
