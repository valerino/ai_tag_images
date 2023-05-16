import json
import logging
from typing import List, Optional

from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

_logger = logging.getLogger()
_processor: BlipProcessor = None
_model: tuple = None


def _load_model(config_path: str = './config.json') -> tuple:
    """
    loads(and possibly downloads to default huggingface models cache) the model set in the configuration. by default, the configuration uses 
        model="Salesforce/blip-image-captioning-large"

    Args:
        config_path (str, optional): _description_. Defaults to './config.json'.

    Returns:
        tuple: _description_
    """
    # load config
    with open(config_path, 'r') as f:
        js = f.read()
        config = json.loads(js)

    # load model
    n = config['captioning']
    model_id = n['model']
    _logger.info('loading model %s ...' % (model_id))

    model = BlipForConditionalGeneration.from_pretrained(model_id)
    processor = BlipProcessor.from_pretrained(model_id)
    global _processor, _model
    _processor = processor
    _model = model
    _logger.info('model%s initialized!' % (model_id))
    return model, processor


def get_img_caption(img_path: str, config_path: str = './config.json') -> str:
    """
    generates a caption using BLIP for the given image.
    if the model is not yet loaded, first time calling this function loads (and possibly downloads) the model

    Args:
        img_path (str): _description_
        config_path (str, optional): _description_. Defaults to './config.json'.

    Returns:
        str: _description_
    """
    global _processor, _model
    if _model is None:
        _load_model(config_path)

    # open image
    _logger.info('opening image %s ...' % (img_path))
    img = Image.open(img_path).convert('RGB')
    inputs = _processor(img, return_tensors="pt")

    # caption
    _logger.info('generating caption for image %s ...' % (img_path))
    d = _model.generate(**inputs)
    desc = _processor.decode(d[0], skip_special_tokens=True)
    return desc
