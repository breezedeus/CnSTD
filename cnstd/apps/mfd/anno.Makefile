MODEL_TYPE = 'yolov7'
MODEL_FP = 'epoch_124-mfd.pt'
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = '/data/jinlong/std_data'
INPUT_IMAGE_DIR = ''

# 生成检测结果（json格式）文件，这个文件可以导入到label studio中，生成待标注的任务
predict:
	python scripts/gen_label_studio_json.py --model-type $(MODEL_TYPE) --model-fp $(MODEL_FP) \
	--resized-shape 608 -l $(LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT)) -i $(INPUT_MODEL_TYPE) -o 'prediction_results.json'


.PHONY: predict
