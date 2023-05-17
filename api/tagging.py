import json
import logging
import os
import time
from tempfile import NamedTemporaryFile
from typing import List, Optional

import mim
from mmdet.apis import DetInferencer
from mmdet.evaluation.functional.class_names import get_classes

_inferencer: DetInferencer = None

_logger = logging.getLogger()


def load_model(config_path: str = './config.json', ) -> DetInferencer:
    """
    loads the model set in the configuration

    Args:
        config_path (str, optional): _description_. Defaults to './config.json'.

    Returns:
        DetInferencer: _description_
    """
    # load config
    with open(config_path, 'r') as f:
        js = f.read()
        config = json.loads(js)

    n = config['tagging']
    cache_dir = os.path.abspath(config.get('models_cache', './models'))

    _logger.info(n)
    model = n['model']
    # assume cpu by default, either use 'cuda:0'
    device = n.get('device', 'cpu')
    model_path = os.path.join(cache_dir, model)

    if not os.path.exists(model_path + '.py'):
        # download model
        _logger.info('downloading model %s ...' % (model_path))
        checkpoints = mim.download('mmdet', [model], dest_root=cache_dir)
        checkpoints_path = os.path.join(cache_dir, checkpoints[0])

        # save weigths back into config to be reused
        n['checkpoints'] = checkpoints_path
        with open(config_path, 'w') as f:
            f.write(json.dumps(config, indent=2))
    else:
        # read from config
        checkpoints_path = n['checkpoints']

    # load model
    _logger.info('loading model %s ...' % (model_path))
    model_path += '.py'
    inferencer = DetInferencer(
        model=model_path, weights=checkpoints_path, device=device)
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
        load_model(config_path)

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
