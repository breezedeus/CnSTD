ROOT_DIR = data
#TRAIN_IDX_FP = data/train.txt
TRAIN_IDX_FP = data/icdar2015/train.txt

BACKBONE = mobilenetv3
EPOCHS = 50
OPTIMIZER = adam
LR = 3e-4
NUM_GPU = -1

train:
	nohup cnstr train --backbone $(BACKBONE) -r $(ROOT_DIR) -i $(TRAIN_IDX_FP) -o ckpt --optimizer $(OPTIMIZER) \
	--epoch $(EPOCHS) --gpu $(NUM_GPU) --batch_size 4 --lr $(LR) > nohup-$(BACKBONE).out 2>&1 &

MAX_SIZE = 768# 640
PSE_THRSH = 0.45
PSE_MIN_AREA = 100

evaluate:
	cnstr evaluate --backbone $(BACKBONE) --model_epoch 40 \
	-i examples -o outputs-$(BACKBONE)-size$(MAX_SIZE)-thrsh$(PSE_THRSH)-area$(PSE_MIN_AREA) \
	--max_size $(MAX_SIZE) --pse_threshold $(PSE_THRSH) --pse_min_area $(PSE_MIN_AREA)

.PHONY: train evaluate
