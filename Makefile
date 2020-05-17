NUM_GPU = 2
#DATA_DIR = data/icdar2015/train
DATA_DIR = /data/cnstr/train
PRETRAIN_MODEL = ~/.mxnet/models/resnet50_v1b-0ecdba34.params
#PRETRAIN_MODEL = ckpt/model_22.param

train:
	nohup python train.py $(DATA_DIR) $(PRETRAIN_MODEL) $(NUM_GPU) > nohup.out 2>&1 &

.PHONY: train
