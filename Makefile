ROOT_DIR = data
TRAIN_IDX_FP = data/icdar2015/train.txt
PRETRAIN_MODEL = ~/.mxnet/models/resnet50_v1b-0ecdba34.params
#PRETRAIN_MODEL = ckpt/model_22.param

NUM_GPU = -1
OPTIMIZER = adam
EPOCHS = 50
LR = 3e-4

MAX_SIZE = 768# 640
PSE_THRSH = 0.45
PSE_MIN_AREA = 100

train:
	nohup cnstr train -i $(DATA_DIR) -o ckpt --pretrain_model_fp $(PRETRAIN_MODEL) --optimizer $(OPTIMIZER) --epoch $(EPOCHS) --gpu $(NUM_GPU) --batch_size 4 --lr $(LR) > nohup.out 2>&1 &

evaluate:
	cnstr evaluate -i examples -o tmp_outputs-size$(MAX_SIZE)-thrsh$(PSE_THRSH)-area$(PSE_MIN_AREA) --model_fp ckpt/model_45.param \
	--max_size $(MAX_SIZE) --pse_threshold $(PSE_THRSH) --pse_min_area $(PSE_MIN_AREA)

.PHONY: train evaluate
