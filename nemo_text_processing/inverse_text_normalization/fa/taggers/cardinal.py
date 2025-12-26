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

from nemo_text_processing.text_normalization.fa.graph_utils import NEMO_SPACE, GraphFst


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals in inverse text normalization.
    Converts spoken form to written form, e.g.
        "صد و بیست و سه" -> cardinal { integer: "123" }
        "منفی پنج" -> cardinal { integer: "-5" }

    Args:
        tn_cardinal: cardinal FST from text normalization (to invert)
    """

    def __init__(self, tn_cardinal):
        super().__init__(name="cardinal", kind="classify")

        # Invert the TN cardinal graph: spoken -> written
        self.graph = pynini.invert(tn_cardinal.cardinal_numbers).optimize()

        # Handle negative numbers: "منفی" -> "-"
        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("منفی", '"-"') + pynutil.delete(NEMO_SPACE),
            0,
            1,
        )

        final_graph = optional_minus_graph + pynutil.insert('integer: "') + self.graph + pynutil.insert('"')

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
