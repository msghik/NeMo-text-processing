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

from nemo_text_processing.text_normalization.fa.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_space,
    insert_space,
)


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measurements, e.g.
        measure { cardinal { integer: "پنجاه" } units: "درصد" } -> "پنجاه درصد"
        measure { cardinal { integer: "صد" } units: "کیلوگرم" } -> "صد کیلوگرم"

    Args:
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, cardinal=None, decimal=None, deterministic: bool = True):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)

        # Cardinal part
        cardinal_part = (
            pynutil.delete("cardinal { integer: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\" }")
        )

        # Decimal part
        decimal_integer = (
            pynutil.delete('integer_part: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        decimal_fractional = (
            pynutil.delete('fractional_part: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        decimal_part = (
            pynutil.delete("decimal { ")
            + decimal_integer
            + pynutil.insert(" ممیز ")
            + delete_space
            + decimal_fractional
            + delete_space
            + pynutil.delete("}")
        )

        # Units
        units = (
            pynutil.delete('units: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # Graphs
        graph_cardinal_units = cardinal_part + delete_space + insert_space + units
        graph_units_cardinal = units + delete_space + insert_space + cardinal_part
        graph_decimal_units = decimal_part + delete_space + insert_space + units

        self.graph = (
            graph_cardinal_units
            | pynutil.add_weight(graph_units_cardinal, 0.1)
            | pynutil.add_weight(graph_decimal_units, 0.1)
        )

        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()
