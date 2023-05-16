import json
import logging
import time
from tempfile import NamedTemporaryFile
from typing import List, Optional

from mmdet.apis import DetInferencer
from mmdet.evaluation.functional.class_names import get_classes

_inferencer: DetInferencer = None

_logger = logging.getLogger()


def _load_model(config_path: str = './config.json', ) -> DetInferencer:
    """
    loads the model set in the configuration. by default, the configuration uses 
        weigths="./models/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth"
        model="./models/rtmdet_x_8xb32-300e_coco.py"
        device="cpu"

        device can be "cuda:0" to use cuda.

    Args:
        config_path (str, optional): _description_. Defaults to './config.json'.

    Returns:
        DetInferencer: _description_
    """
    # load config
    with open(config_path, 'r') as f:
        js = f.read()
        config = json.loads(js)

    # load model
    n = config['tagging']
    _logger.info(n)
    model = n['model_config']
    weights_path = n['weights']
    device = n['device']

    inferencer = DetInferencer(
        model=model, weights=weights_path, device=device)
    global _inferencer
    _inferencer = inferencer

    return inferencer


def _arrange_tags(tags: list) -> dict:
    """
    arrange tags in a dict with tag counts, like { "tag1": n, "tag2": m, "tag3": o, ...}

    Args:
        tags (list): _description_

    Returns:
        dict: _description_
    """
    d = {}
    for t in tags:
        # check how many t in tags
        count = 0
        for tt in tags:
            if tt == t:
                count += 1

        d[t] = count

    return d


def get_img_tags(img_path: str, config_path: str = './config.json', threshold: float = 0.7) -> dict:
    """
    generate tags for the given image, optionally specifying a detection threshold (default=0.7, fair good)
    if the model is not yet loaded, first time calling this function loads the model

    Args:
        img_path (str): _description_
        config_path (str, optional): _description_. Defaults to './config.json'.
        threshold (float, optional): _description_. Defaults to 0.7.

    Returns:
        dict: _description_
    """
    global _inferencer
    if _inferencer is None:
        _load_model(config_path)

    # inference
    _logger.info('getting tags for image=%s, threshold=%f' % (
        img_path, threshold))
    result = _inferencer(
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

    tags = _arrange_tags(tags)
    _logger.info('tags=%s ...' % (tags))
    return tags
