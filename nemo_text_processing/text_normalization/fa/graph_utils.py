# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# Copyright 2015 and onwards Google, Inc.
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

import string

# Import base classes and common utilities from English module (as per CONTRIBUTING.md)
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_CHAR,
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    NEMO_NOT_SPACE,
    NEMO_PUNCT,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_extra_space,
    delete_preserve_order,
    delete_space,
    delete_zero_or_one_space,
    generator_main,
    insert_space,
)

try:
    import pynini
    from pynini.lib import pynutil

    # Persian-specific characters
    FA_ALPHA = pynini.union(
        "آ",
        "ا",
        "ب",
        "پ",
        "ت",
        "ث",
        "ج",
        "چ",
        "ح",
        "خ",
        "د",
        "ذ",
        "ر",
        "ز",
        "ژ",
        "س",
        "ش",
        "ص",
        "ض",
        "ط",
        "ظ",
        "ع",
        "غ",
        "ف",
        "ق",
        "ک",
        "گ",
        "ل",
        "م",
        "ن",
        "و",
        "ه",
        "ی",
        "ئ",
        "ء",
        "ة",
    ).optimize()

    # Persian digits (۰-۹)
    FA_DIGIT = pynini.union("۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹").optimize()

    # Mapping from Persian digits to ASCII digits
    FA_TO_ASCII_DIGIT = pynini.string_map(
        [
            ("۰", "0"),
            ("۱", "1"),
            ("۲", "2"),
            ("۳", "3"),
            ("۴", "4"),
            ("۵", "5"),
            ("۶", "6"),
            ("۷", "7"),
            ("۸", "8"),
            ("۹", "9"),
        ]
    ).optimize()

    # Persian conjunction "و" (and)
    insert_va = pynutil.insert(" و ")

    PYNINI_AVAILABLE = True

except (ModuleNotFoundError, ImportError):
    FA_ALPHA = None
    FA_DIGIT = None
    FA_TO_ASCII_DIGIT = None
    insert_va = None

    PYNINI_AVAILABLE = False
