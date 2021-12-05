MODEL_NAME = db_resnet18

train:
	cnstd train -m $(MODEL_NAME) --train-config-fp examples/train_config.json -i data/icdar2015

predict:
	cnstd predict -m $(MODEL_NAME) --model_epoch 29 --rotated-bbox --box-score-thresh 0.3 --resized-shape 768,768 \
	--context cpu -i examples -o prediction

demo:
	pip install streamlit
	streamlit run cnstd/app.py

package:
	python setup.py sdist bdist_wheel

VERSION = 1.1.1
upload:
	python -m twine upload  dist/cnstd-$(VERSION)* --verbose


.PHONY: train predict demo package upload
