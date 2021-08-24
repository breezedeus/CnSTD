MODEL_NAME = db_resnet18

train:
	cnstd train -m $(MODEL_NAME) --train-config-fp examples/train_config_gpu.json -i data 

predict:
	cnstd predict -m $(MODEL_NAME) --model_epoch 29 --rotated-bbox --box-score-thresh 0.3 --resized-shape 768,768 \
	--context cuda:0 -i examples -o prediction

package:
	python setup.py sdist bdist_wheel

VERSION = 1.0.0
upload:
	python -m twine upload  dist/cnstd-$(VERSION)* --verbose


.PHONY: train predict package upload
