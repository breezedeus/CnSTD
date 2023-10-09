MODEL_NAME = db_resnet18

train:
	cnstd train -m $(MODEL_NAME) --train-config-fp examples/train_config.json -i data/icdar2015

predict:
	cnstd predict -m $(MODEL_NAME) --model_epoch 29 --rotated-bbox --box-score-thresh 0.3 --resized-shape 768,768 \
	--context cpu -i examples -o prediction

layout:
	cnstd analyze -m layout --conf-thresh 0.25 --resized-shape 800 --img-fp examples/mfd/zh.jpg

mfd:
	cnstd analyze -m mfd --conf-thresh 0.25 --resized-shape 700 --img-fp examples/mfd/zh4.jpg

demo:
	pip install streamlit
	streamlit run cnstd/app.py

package:
	rm -rf build
	python setup.py sdist bdist_wheel

VERSION = 1.2.3.5
upload:
	python -m twine upload  dist/cnstd-$(VERSION)* --verbose


.PHONY: train predict layout mfd demo package upload
