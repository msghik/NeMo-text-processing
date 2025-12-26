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

from nemo_text_processing.text_normalization.fa.graph_utils import NEMO_SPACE, GraphFst, insert_space


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimals in inverse text normalization.
    Converts spoken form to written form, e.g.
        "سه ممیز چهارده صدم" -> decimal { integer_part: "3" fractional_part: "14" }
        "منفی دو ممیز پنج دهم" -> decimal { negative: "-" integer_part: "2" fractional_part: "5" }

    Args:
        tn_decimal: decimal FST from text normalization (to invert)
    """

    def __init__(self, tn_decimal):
        super().__init__(name="decimal", kind="classify")

        # Invert the TN decimal components
        integer_graph = pynini.invert(tn_decimal.integer_part).optimize()
        fractional_graph = pynini.invert(tn_decimal.graph_fractional).optimize()

        # Handle negative: "منفی" -> "-"
        optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("منفی", '"-"') + pynutil.delete(NEMO_SPACE),
            0,
            1,
        )

        # Persian decimal separator word: ممیز
        decimal_separator = pynutil.delete(" ممیز ")

        graph_integer = pynutil.insert('integer_part: "') + integer_graph + pynutil.insert('"')
        graph_fractional = pynutil.insert('fractional_part: "') + fractional_graph + pynutil.insert('"')

        final_graph = optional_minus_graph + graph_integer + decimal_separator + insert_space + graph_fractional

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
