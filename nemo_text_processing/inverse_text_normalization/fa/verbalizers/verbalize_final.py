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

import pynini
from pynini.lib import pynutil

from nemo_text_processing.inverse_text_normalization.fa.verbalizers.verbalize import VerbalizeFst
from nemo_text_processing.text_normalization.fa.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_space,
)


class VerbalizeFinalFst(GraphFst):
    """
    Finite state transducer that verbalizes an entire sentence for Persian ITN.
    Combines the VerbalizeFst with token handling for complete sentence processing.
    """

    def __init__(self):
        super().__init__(name="verbalize_final", kind="verbalize")

        verbalize = VerbalizeFst().fst

        # Word graph for handling plain words
        word = (
            pynutil.delete("name:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        types = verbalize | word

        token = pynutil.delete("tokens { ") + types + pynutil.delete(" }")
        graph = token + pynini.closure(delete_extra_space + token)
        graph = delete_space + graph + delete_space

        self.fst = graph.optimize()
