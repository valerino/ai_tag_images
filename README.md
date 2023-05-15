# tag_image

image tagging through RTMDet model with COCO weights, using [mmdetection](https://github.com/open-mmlab/mmdetection) framework.

## install

requires python 3.9+

~~~bash
pip3 install -r ./requirements
git clone https://github.com/open-mmlab/mmdetection
pip3 install -e ./mmdetection

# download the rtmdet extra-large coco model (~350MB)
mim download mmdet --config rtmdet_x_8xb32-300e_coco --dest ./models
~~~

## usage

~~~bash
./tag_image.py --help
usage: tag_image.py [-h] [--img IMG] [--config CONFIG] [--weights WEIGHTS]
                    [--device DEVICE] [--threshold THRESHOLD]
                    [--server SERVER] [--remove_duplicate_tags]

tags images using RTMDet model with COCO weights.

options:
  -h, --help            show this help message and exit
  --img IMG             path to the input image, ignored if --server is
                        specified.
  --config CONFIG       path to model configuration .py file, may be None
                        to extract from weights .pth file,
                        default="./models/rtmdet_x_8xb32-300e_coco.py"
  --weights WEIGHTS     path to the weights(checkpoints) file .pth, default
                        ="./models/rtmdet_x_8xb32-
                        300e_coco_20220715_230555-cc79b9ae.pth"
  --device DEVICE       device used for inference, default="cpu", use
                        "cuda:0" for gpu.
  --threshold THRESHOLD
                        detection threshold, discard results with score <
                        threshold, default=0.7f
  --server SERVER       start server on the given iface:port, if set --img
                        is ignored
  --remove_duplicate_tags
                        remove duplicate tags, if any. ignored if --server
                        is specified.
~~~

### commandline inference

~~~bash
./tag_image.py --threshold 0.5 --img ./sample_images/tennis.jpg
....
{'img': './sample_images/tennis.jpg', 'tags': ['person', 'person', 'sports_ball', 'tennis_racket']}
~~~

### inference through REST api

using REST api server, start server first

~~~bash
# http://localhost:8080/docs for docs
./tag_image.py --server 0.0.0.0:8080 
~~~

then call *tag_image* endpoint

~~~bash
curl -L -F "threshold=0.5" -F "remove_duplicates=False" -F "job_id=prova" -F "file=@./sample_images/tennis.jpg" http://127.0.0.1:8080/tag_image
{"status":"success","time_msec":1684185690464,"data":{"tags":["person","person","sports_ball","tennis_racket"]},"req_id":"1684185689243813474","job_id":"prova"}
~~~


