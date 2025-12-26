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

from nemo_text_processing.inverse_text_normalization.fa.utils import get_abs_path
from nemo_text_processing.text_normalization.fa.graph_utils import GraphFst, insert_space


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fractions in inverse text normalization.
    Converts spoken form to written form, e.g.
        "یک دوم" -> fraction { numerator: "1" denominator: "2" }
        "سه چهارم" -> fraction { numerator: "3" denominator: "4" }
        "دو و یک دوم" -> fraction { integer_part: "2" numerator: "1" denominator: "2" }

    Args:
        tn_cardinal: cardinal FST from text normalization (to invert)
    """

    def __init__(self, tn_cardinal):
        super().__init__(name="fraction", kind="classify")

        # Invert cardinal for numerator
        cardinal_graph = pynini.invert(tn_cardinal.cardinal_numbers).optimize()

        # Denominator mapping (spoken ordinal -> digit)
        denominator_map = pynini.string_map(
            [
                ("دوم", "2"),
                ("سوم", "3"),
                ("چهارم", "4"),
                ("پنجم", "5"),
                ("ششم", "6"),
                ("هفتم", "7"),
                ("هشتم", "8"),
                ("نهم", "9"),
                ("دهم", "10"),
            ]
        )

        graph_numerator = pynutil.insert('numerator: "') + cardinal_graph + pynutil.insert('"')
        graph_denominator = pynutil.insert('denominator: "') + denominator_map + pynutil.insert('"')

        # Fraction: numerator + space + denominator
        graph_fraction = graph_numerator + pynutil.delete(" ") + insert_space + graph_denominator

        # Optional integer part: "دو و یک دوم" -> integer_part: "2" numerator: "1" denominator: "2"
        graph_integer = pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')
        optional_integer = pynini.closure(
            graph_integer + pynutil.delete(" و ") + insert_space,
            0,
            1,
        )

        final_graph = optional_integer + graph_fraction

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
