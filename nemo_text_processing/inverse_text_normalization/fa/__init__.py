# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Persian (Farsi) Inverse Text Normalization module.

This module converts spoken form text to written form, e.g.:
- "صد و بیست و سه" -> "123"
- "سه ممیز چهارده صدم" -> "3.14"
- "ساعت سه و سی دقیقه" -> "3:30"
- "صد دلار" -> "$100"
- "پنجاه درصد" -> "50%"
"""

from nemo_text_processing.inverse_text_normalization.fa.taggers.tokenize_and_classify import ClassifyFst
from nemo_text_processing.inverse_text_normalization.fa.verbalizers.verbalize import VerbalizeFst
from nemo_text_processing.inverse_text_normalization.fa.verbalizers.verbalize_final import VerbalizeFinalFst

__all__ = ["ClassifyFst", "VerbalizeFst", "VerbalizeFinalFst"]
from nemo_text_processing.utils.logging import logger

try:
    import pynini

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    logger.warning(
        "`pynini` is not installed ! \n"
        "Please run the `nemo_text_processing/setup.sh` script"
        "prior to usage of this toolkit."
    )
