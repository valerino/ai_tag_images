#!/usr/bin/env python3

# prerequisites:
#
# pip3 install -r ./requirements
# git clone https://github.com/open-mmlab/mmdetection
# pip3 install -e ./mmdetection

#
# to download model:
# mim download mmdet --config rtmdet_x_8xb32-300e_coco --dest ./models
#

import json
import logging
import os
from argparse import ArgumentParser

import api.captioning
import api.rest
import api.tagging
import coloredlogs

_logger = logging.getLogger()
coloredlogs.install(level='INFO')


def main():
    parser = ArgumentParser(
        description='tags images using RTMDet model with COCO weights, and add caption using BLIP.')
    parser.add_argument(
        '--img_path', nargs=1, type=str, help='path to the input image, ignored if --server is specified.')
    parser.add_argument('--config_path', type=str, nargs=1, default=['./config.json'],
                        help='path to the configuration json')
    parser.add_argument('--tag_threshold', type=float, nargs=1,
                        default=[0.7], help='tagging_detection threshold, discard results with score < threshold, default=0.7f')
    parser.add_argument(
        '--server', help='start server on the given iface:port, if set --img is ignored', type=str, nargs=1, default=[None])
    parser.add_argument(
        '--add_caption', help='also generate caption for the image. ignored if --server is specified.', action='store_const', const=True, default=False)
    args = parser.parse_args()

    try:
        if args.server[0] is None:
            # tag
            _logger.info('[.] command line, img=%s, config_path=%s, threshold=%s, add_caption=%r ...' % (
                args.img_path[0], args.config_path[0], args.tag_threshold[0], args.add_caption))
            tags = api.tagging.get_img_tags(img_path=args.img_path[0], config_path=args.config_path[0],
                                            threshold=args.tag_threshold[0])
            caption = None
            if args.add_caption:
                # caption too
                caption = api.captioning.get_img_caption(
                    img_path=args.img_path[0], config_path=args.config_path[0])

            js = api.rest.build_result(
                tags=tags, caption=caption, file_name=os.path.basename(args.img_path[0]))
            print(js)
        else:
            # start server
            api.rest.start_server(args)

    except Exception as ex:
        _logger.exception(ex)
        return 1

    return 0


if __name__ == '__main__':
    main()
