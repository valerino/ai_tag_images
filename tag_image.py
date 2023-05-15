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

import api.rest
import coloredlogs

_logger = logging.getLogger()
coloredlogs.install(level='INFO')


def main():
    parser = ArgumentParser(description='tags images using RTMDet model with COCO weights.')
    parser.add_argument(
        '--img', nargs=1, type=str, help='path to the input image, ignored if --server is specified.')
    parser.add_argument('--config', type=str, nargs=1, default=['./models/rtmdet_x_8xb32-300e_coco.py'],
                        help='path to model configuration .py file, may be None to extract from weights .pth file, default="./models/rtmdet_x_8xb32-300e_coco.py"')
    parser.add_argument('--weights', nargs=1, help='path to the weights(checkpoints) file .pth, default="./models/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth"',
                        default=['./models/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth'])
    parser.add_argument(
        '--device', default=['cpu'], help='device used for inference, default="cpu", use "cuda:0" for gpu.', nargs=1)
    parser.add_argument('--threshold', type=float, nargs=1,
                        default=[0.7], help='detection threshold, discard results with score < threshold, default=0.7f')
    parser.add_argument(
        '--server', help='start server on the given iface:port, if set --img is ignored', type=str, nargs=1, default=[None])
    parser.add_argument(
        '--remove_duplicate_tags', help='remove duplicate tags, if any. ignored if --server is specified.', action='store_const', const=True, default=False)
    args = parser.parse_args()

    try:
        if args.server[0] is None:
            _logger.info('[.] command line inference, img=%s, model=%s, weights=%s, device=%s, threshold=%s, remove_duplicate_tags=%r ...' % (
                args.img[0], args.config[0], args.weights[0], args.device[0], args.threshold[0], args.remove_duplicate_tags))
            tags = api.rest.get_img_tags(img_path=args.img[0], weights_path=args.weights[0], config_path=args.config[0], device=args.device[0],
                                         threshold=args.threshold[0], remove_duplicates=args.remove_duplicate_tags)
            js = {"img": args.img[0], "tags": tags}
            print(js)
            print("done!")
        else:
            # start server
            api.rest.start_server(args)

    except Exception as ex:
        _logger.exception(ex)
        return 1

    return 0


if __name__ == '__main__':
    main()
