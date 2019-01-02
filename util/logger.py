#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created by luozhenyu on 2018/12/10
"""
import os
import logging.config
from datetime import datetime, timedelta, timezone


def get_log_dict(log_dir, log_file):
    log_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "[%(asctime)s][%(name)s:%(levelname)s][%(module)s:%(funcName)s]:%(message)s"
            }
        },

        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "standard"
            },

            "log": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "standard",
                "filename": os.path.join(log_dir, log_file),
                "maxBytes": 10485760,
                "backupCount": 5,
                "encoding": "utf8"
            }
        },

        "loggers": {
            "default": {
                "level": "INFO",
                "handlers": ["console", "log"],
                "propagate": True
            }
        }
    }
    return log_dict


def init_log_conf():
    bj_time = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=8)))
    base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
    log_dir = os.path.join(base_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file = bj_time.strftime("%Y%m%d-%H%M%S") + ".log"
    log_dict = get_log_dict(log_dir, log_file)
    logging.config.dictConfig(log_dict)
    log = logging.getLogger("default")
    return log


logger = init_log_conf()
