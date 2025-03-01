# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

# configure a global logger with a module-specific prefix
log = logging.getLogger("meta.flowdec")
log.setLevel(logging.INFO)
