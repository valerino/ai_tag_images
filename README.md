# tag_image

AI image tagging and captioning, using [mmdetection](https://github.com/open-mmlab/mmdetection) framework and BLIP.

## install

requires python 3.9+

~~~bash
pip3 install -r ./requirements
git clone https://github.com/open-mmlab/mmdetection
pip3 install -e ./mmdetection

# download the rtmdet extra-large coco model (~350MB) used for image tagging with mmdetection
mim download mmdet --config rtmdet_x_8xb32-300e_coco --dest ./models
~~~

## usage

~~~bash
./tag_image.py --help
usage: tag_image.py [-h] [--img_path IMG_PATH] [--config_path CONFIG_PATH]
                    [--tag_threshold TAG_THRESHOLD] [--server SERVER]
                    [--add_caption]

tags images using RTMDet model with COCO weights, and add caption using
BLIP.

options:
  -h, --help            show this help message and exit
  --img_path IMG_PATH   path to the input image, ignored if --server is
                        specified.
  --config_path CONFIG_PATH
                        path to the configuration json
  --tag_threshold TAG_THRESHOLD
                        tagging_detection threshold, discard results with
                        score < threshold, default=0.7f
  --server SERVER       start server on the given iface:port, if set --img
                        is ignored
  --add_caption         also generate caption for the image. ignored if
                        --server is specified.
~~~

### configuration

the [default configuration](./config.json) can be edited to use other models.

> the models are loaded (and possibly downloaded, for the captioning part) the first time the image processing code is called. so, first call will always take more.

### commandline

~~~bash
./tag_image.py --tag_threshold 0.5 --img_path ./sample_images/tennis.jpg --add_caption
....
{'status': 'success', 'time_msec': 1684269941755, 'data': {'tags': {'person': 2, 'sports_ball': 1, 'tennis_racket': 1}, 'caption': 'two pictures of a man holding a trophy and a tennis ball', 'file_name': 'tennis.jpg'}}
~~~

### through REST api

using REST api server, start server first

~~~bash
# http://localhost:8080/docs for docs
./tag_image.py --server 0.0.0.0:8080 
~~~

then call *process_image* endpoint

~~~bash
curl -L -F "threshold=0.5" -F "add_caption=True" -F "job_id=prova" -F "file_name=ducks.jpg" -F "file=@./sample_images/ducks.jpg" http://127.0.0.1:8080/process_image
{"status":"success","time_msec":1684269861711,"data":{"tags":{"bird":2},"file_name": "ducks.jpg", "caption":"ducks standing on a rock near the water and a river"},"req_id":"1684269859207291701","job_id":"prova"}
~~~
