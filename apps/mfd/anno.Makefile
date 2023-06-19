MODEL_TYPE = 'yolov7'
MODEL_FP = '/home/ein/.cnstd/1.2/analysis/mfd-yolov7-epoch224-20230613.pt'
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = '/data/jinlong/std_data'
INPUT_IMAGE_DIR = '/data/jinlong/std_data/call_images/images/2023-02-27_2023-03-05'

# 生成检测结果（json格式）文件，这个文件可以导入到label studio中，生成待标注的任务
predict:
	python scripts/gen_label_studio_json.py --model-type $(MODEL_TYPE) --model-fp $(MODEL_FP) \
	--resized-shape 608 -l $(LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT) -i $(INPUT_IMAGE_DIR) -o 'prediction_results.json'

convert_to_yolov7:
	python scripts/convert_label_studio_to_yolov7.py --anno-json-fp-list 'prediction_results.json' \
	--index-prefix 'data/call_images/images/2023-02-27_2023-03-05' \
	--out-labels-dir 'data/call_images/labels/2023-02-27_2023-03-05' --out-index-fp 'train.txt'

.PHONY: predict convert_to_yolov7
