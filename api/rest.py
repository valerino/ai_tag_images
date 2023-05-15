import logging
import os
import random
import time
from tempfile import NamedTemporaryFile
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from mmdet.apis import DetInferencer
from mmdet.evaluation.functional.class_names import get_classes

_logger = logging.getLogger()
_inferencer: DetInferencer = None
_app = FastAPI()


def _load_model(weights_path: str = './models/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth',
                config_path: str = './models/rtmdet_x_8xb32-300e_coco.py', device: str = 'cpu') -> DetInferencer:
    """_summary_

    Args:
        weights_path (str, optional): _description_. Defaults to './models/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth'.
        config_path (str, optional): _description_. Defaults to './models/rtmdet_x_8xb32-300e_coco.py'.
        device (str, optional): _description_. Defaults to 'cpu'.

    Returns:
        DetInferencer: _description_
    """
    # load model
    inferencer = DetInferencer(
        model=config_path, weights=weights_path, device=device)
    return inferencer


def get_img_tags(img_path: str, inferencer: DetInferencer = None, weights_path: str = './models/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth', config_path: str = './models/rtmdet_x_8xb32-300e_coco.py', device: str = 'cpu', threshold: float = 0.7, remove_duplicates: bool = False) -> List:
    """_summary_

    Args:
        img_path (str): _description_
        inferencer (DetInferencer, optional): _description_. Defaults to None.
        weights_path (str, optional): _description_. Defaults to './models/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth'.
        config_path (str, optional): _description_. Defaults to './models/rtmdet_x_8xb32-300e_coco.py'.
        device (str, optional): _description_. Defaults to 'cpu'.
        threshold (float, optional): _description_. Defaults to 0.7.
        remove_duplicates (bool, optional): _description_. Defaults to False.

    Returns:
        List: _description_
    """
    if inferencer is None:
        # load model
        inferencer = _load_model(
            weights_path=weights_path, config_path=config_path, device=device)

    # inference
    _logger.info('getting tags for image=%s, threshold=%f, remove duplicate tags=%r ...' % (
        img_path, threshold, remove_duplicates))
    result = inferencer(
        inputs=img_path, batch_size=1, no_save_vis=True, pred_score_thr=threshold)

    class_names = get_classes('coco')
    _logger.info('[.] inferencing against %d classes ...' % (len(class_names)))
    predictions = result['predictions'][0]
    labels = predictions['labels']
    scores = predictions['scores']

    tags = []
    for i, label in enumerate(labels):
        if scores[i] > threshold:
            tags.append(class_names[label])

    if remove_duplicates:
        _logger.warning(
            'removing duplicate tags if any, original tags=%s ...' % (tags))
        tags = list(dict.fromkeys(tags))

    _logger.info('tags=%s ...' % (tags))
    return tags


@_app.post("/tag_image")
async def tag_image_handler(file: UploadFile = File(...), threshold: float = Form(default=0.7), remove_duplicates: bool = Form(default=False),
                            req_id: str = Form(default=None), job_id: str = Form(default=None)) -> dict:
    """_summary_

    Args:
        file (UploadFile, optional): _description_. Defaults to File(...).
        threshold (float, optional): _description_. Defaults to Form(...).
        remove_duplicates (bool, optional): _description_. Defaults to Form(...).
        req_id (str, optional): _description_. Defaults to Form(...).
        job_id (str, optional): _description_. Defaults to Form(...).

    Returns:
        dict: _description_
    """

    # get parameters
    temp_file = NamedTemporaryFile(delete=False)
    if req_id is None:
        # calculate random req_id
        t = time.time_ns() + random.randint(1, 64000) + random.randint(1, 64000)
        req_id = str(t)

    # dump file to temp
    try:
        contents = file.file.read()
        with temp_file as f:
            f.write(contents)
    except Exception as ex:
        os.unlink(temp_file.name)
        raise HTTPException(
            status_code=500, detail='error in file upload!') from ex
    finally:
        file.file.close()

    # inference
    tags = []
    try:
        tags = get_img_tags(img_path=temp_file.name, inferencer=_inferencer,
                            threshold=threshold,  remove_duplicates=remove_duplicates)
    except Exception as ex:
        raise HTTPException(
            status_code=500, detail='inference error !') from ex
    finally:
        os.unlink(temp_file.name)

    #  done
    now = time.time_ns() // 1_000_000
    js = {'status': 'success', 'time_msec':  now,
          'data': {'tags': tags}, 'req_id': req_id}

    if job_id is not None:
        js['job_id'] = job_id
    return js


def start_server(args: dict):
    """_summary_

    Args:
        args (dict): _description_
    """
    # load model
    global _inferencer
    _inferencer = _load_model(args.weights[0], args.config[0], args.device[0])

    # start server
    _logger.info('[.] starting server at %s ...' % (args.server[0]))
    splitted = args.server[0].split(':')
    uvicorn.run(_app, host=splitted[0], port=int(splitted[1]))
