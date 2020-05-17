NUM_GPU = -1
#DATA_DIR = data/icdar2015/train
DATA_DIR = /data/cnstr/train
PRETRAIN_MODEL = ~/.mxnet/models/resnet50_v1b-0ecdba34.params
#PRETRAIN_MODEL = ckpt/model_22.param

train:
	nohup cnstr train -i $(DATA_DIR) -o ckpt --pretrain_model_fp $(PRETRAIN_MODEL) --epoch 50 --gpu $(NUM_GPU) --batch_size 4 --lr 1e-3 > nohup.out 2>&1 &

.PHONY: train
