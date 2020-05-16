NUM_GPU = 2
PRETRAIN_MODEL = ~/.mxnet/models/resnet50_v1b-0ecdba34.params

train:
	python train.py data/icdar2015/train $(PRETRAIN_MODEL) $(NUM_GPU)

.PHONY: train
