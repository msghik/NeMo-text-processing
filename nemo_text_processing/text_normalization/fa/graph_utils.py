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

import os
import string
from pathlib import Path
from typing import Dict

from nemo_text_processing.text_normalization.en.utils import get_abs_path
from nemo_text_processing.utils.logging import logger

try:
    import pynini
    from pynini import Far
    from pynini.examples import plurals
    from pynini.export import export
    from pynini.lib import byte, pynutil, utf8

    NEMO_CHAR = utf8.VALID_UTF8_CHAR

    NEMO_DIGIT = byte.DIGIT
    NEMO_LOWER = pynini.union(*string.ascii_lowercase).optimize()
    NEMO_UPPER = pynini.union(*string.ascii_uppercase).optimize()
    NEMO_ALPHA = pynini.union(NEMO_LOWER, NEMO_UPPER).optimize()
    NEMO_ALNUM = pynini.union(NEMO_DIGIT, NEMO_ALPHA).optimize()
    NEMO_HEX = pynini.union(*string.hexdigits).optimize()
    NEMO_NON_BREAKING_SPACE = "\u00a0"
    NEMO_SPACE = " "
    NEMO_WHITE_SPACE = pynini.union(" ", "\t", "\n", "\r", "\u00a0").optimize()
    NEMO_NOT_SPACE = pynini.difference(NEMO_CHAR, NEMO_WHITE_SPACE).optimize()
    NEMO_NOT_QUOTE = pynini.difference(NEMO_CHAR, r'"').optimize()

    NEMO_PUNCT = pynini.union(*map(pynini.escape, string.punctuation)).optimize()
    NEMO_GRAPH = pynini.union(NEMO_ALNUM, NEMO_PUNCT).optimize()

    NEMO_SIGMA = pynini.closure(NEMO_CHAR)

    # Persian-specific characters
    FA_ALPHA = pynini.union(
        "آ", "ا", "ب", "پ", "ت", "ث", "ج", "چ", "ح", "خ",
        "د", "ذ", "ر", "ز", "ژ", "س", "ش", "ص", "ض", "ط",
        "ظ", "ع", "غ", "ف", "ق", "ک", "گ", "ل", "م", "ن",
        "و", "ه", "ی", "ئ", "ء", "ة"
    ).optimize()

    # Persian digits (۰-۹)
    FA_DIGIT = pynini.union("۰", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹").optimize()

    # Mapping from Persian digits to ASCII digits
    FA_TO_ASCII_DIGIT = pynini.string_map([
        ("۰", "0"), ("۱", "1"), ("۲", "2"), ("۳", "3"), ("۴", "4"),
        ("۵", "5"), ("۶", "6"), ("۷", "7"), ("۸", "8"), ("۹", "9"),
    ]).optimize()

    delete_space = pynutil.delete(pynini.closure(NEMO_WHITE_SPACE))
    delete_zero_or_one_space = pynutil.delete(pynini.closure(NEMO_WHITE_SPACE, 0, 1))
    insert_space = pynutil.insert(" ")
    # Persian conjunction "و" (and)
    insert_va = pynutil.insert("و ")
    delete_extra_space = pynini.cross(pynini.closure(NEMO_WHITE_SPACE, 1), " ")
    delete_preserve_order = pynini.closure(
        pynutil.delete(" preserve_order: true")
        | (pynutil.delete(' field_order: "') + NEMO_NOT_QUOTE + pynutil.delete('"'))
    )

    TO_LOWER = pynini.union(*[pynini.cross(x, y) for x, y in zip(string.ascii_uppercase, string.ascii_lowercase)])
    TO_UPPER = pynini.invert(TO_LOWER)

    PYNINI_AVAILABLE = True

except (ModuleNotFoundError, ImportError):
    # Create placeholders for when pynini is not available
    NEMO_CHAR = None
    NEMO_DIGIT = None
    NEMO_SIGMA = None
    NEMO_SPACE = " "
    NEMO_WHITE_SPACE = None
    NEMO_NOT_SPACE = None
    NEMO_NOT_QUOTE = None
    FA_ALPHA = None
    FA_DIGIT = None
    FA_TO_ASCII_DIGIT = None

    delete_space = None
    delete_zero_or_one_space = None
    insert_space = None
    insert_va = None
    delete_extra_space = None

    PYNINI_AVAILABLE = False


class GraphFst:
    """
    Base class for all grammar fsts.

    Args:
        name: name of grammar class
        kind: either 'classify' or 'verbalize'
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, name: str, kind: str, deterministic: bool = True):
        self.name = name
        self.kind = kind
        self._fst = None
        self.deterministic = deterministic

        self.far_path = Path(os.path.dirname(__file__) + "/grammars/" + kind + "/" + name + ".far")
        if self.far_exist():
            self._fst = Far(self.far_path, mode="r", arc_type="standard", far_type="default").get_fst()

    def far_exist(self) -> bool:
        """
        Returns true if FAR can be loaded
        """
        return self.far_path.exists()

    @property
    def fst(self):
        return self._fst

    @fst.setter
    def fst(self, fst):
        self._fst = fst


def generator_main(file_name: str, graphs: Dict[str, pynini.FstLike]):
    """
    Exports graph as OpenFst finite state archive (FAR) file with given file name and target graphs.
    Args:
        file_name: path to export FAR file to
        graphs: dictionary of graph names mapped to graph objects
    """
    exporter = export.Exporter(file_name)
    for name, graph in graphs.items():
        exporter[name] = graph.optimize()
    exporter.close()
    logger.info(f"Created {file_name}")
