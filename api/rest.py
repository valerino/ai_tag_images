import logging
import os
import random
import time
from tempfile import NamedTemporaryFile
from typing import List, Optional

import api.captioning
import api.tagging
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

_logger = logging.getLogger()
_args = {}
_app = FastAPI()


def build_result(tags: dict, caption: str = None, file_name: str = None, req_id: str = None, job_id: str = None) -> dict:
    """
    build a "success" jsend, with optional request id and job_id (added only if provided)

    Args:
        tags (dict): _description_
        file_name (str): _description_. Defaults to None.
        caption (str): _description_. Defaults to None.
        req_id (str, optional): _description_. Defaults to None.
        job_id (str, optional): _description_. Defaults to None.

    Returns:
        dict: _description_
    """
    #  done
    now = time.time_ns() // 1_000_000
    js = {'status': 'success', 'time_msec': now,
          'data': {'tags': tags}}

    if caption is not None:
        js['data']['caption'] = caption

    if file_name is not None:
        js['data']['file_name'] = file_name

    if req_id is not None:
        js['req_id'] = req_id

    if job_id is not None:
        js['job_id'] = job_id
    return js


@_app.post("/process_image")
async def process_image_handler(file: UploadFile = File(...), file_name: str = Form(default=None), threshold: float = Form(default=0.7), add_caption: bool = Form(default=False),
                                req_id: str = Form(default=None), job_id: str = Form(default=None)) -> dict:
    """
    generate tags for the given image, with the given detection threshold (default=0.7, fair good), optionally generates a caption using BLIP

    Args:
        file (UploadFile, optional): _description_. Defaults to File(...).
        file_name (str, optional): _description_. Defaults to Form(default=None).
        threshold (float, optional): _description_. Defaults to Form(default=0.7).
        add_caption (bool, optional): _description_. Defaults to Form(default=False).
        req_id (str, optional): _description_. Defaults to Form(default=None).
        job_id (str, optional): _description_. Defaults to Form(default=None).

    Raises:
        HTTPException: _description_

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

    # inference and captioning
    tags = {}
    desc = None
    try:
        global _args
        config_path = _args.config_path[0]
        tags = api.tagging.get_img_tags(img_path=temp_file.name, config_path=config_path,
                                        threshold=threshold)
        if add_caption:
            # add caption
            desc = api.captioning.get_img_caption(
                img_path=temp_file.name, config_path=config_path)
    except Exception as ex:
        raise HTTPException(
            status_code=500, detail='processing error !') from ex
    finally:
        os.unlink(temp_file.name)

    #  done
    js = build_result(
        tags, caption=desc, file_name=file_name, req_id=req_id, job_id=job_id)
    return js


def start_server(args: dict):
    """
    start REST api server

    Args:
        args (dict): the commandline args dict
    """

    global _args
    _args = args

    # start server
    _logger.info('[.] starting server at %s ...' % (args.server[0]))
    splitted = args.server[0].split(':')
    uvicorn.run(_app, host=splitted[0], port=int(splitted[1]))
