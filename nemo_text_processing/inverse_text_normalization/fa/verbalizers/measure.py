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

from nemo_text_processing.text_normalization.fa.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measurements in ITN.
        e.g. measure { cardinal { integer: "50" } units: "%" } -> "50%"
        e.g. measure { cardinal { integer: "100" } units: "kg" } -> "100 kg"
    """

    def __init__(self):
        super().__init__(name="measure", kind="verbalize")

        # Cardinal component
        cardinal_graph = (
            pynutil.delete("cardinal {")
            + delete_space
            + pynutil.delete("integer:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
            + delete_space
            + pynutil.delete("}")
        )

        # Units
        units_graph = (
            pynutil.delete("units:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # Units that don't need space before them
        no_space_units = pynini.union("%", "°C", "°F")

        # Combine: cardinal + space + units (or no space for special units)
        graph = cardinal_graph + pynutil.insert(" ") + delete_space + units_graph

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
