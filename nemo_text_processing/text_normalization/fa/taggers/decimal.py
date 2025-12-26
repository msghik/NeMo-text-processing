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

from nemo_text_processing.text_normalization.fa.graph_utils import NEMO_DIGIT, GraphFst, insert_space
from nemo_text_processing.text_normalization.fa.utils import get_abs_path


class DecimalFst(GraphFst):
    """
    Finite state transducer for classifying decimal numbers, e.g.
        "3.14" -> decimal { integer_part: "سه" fractional_part: "چهارده صدم" }
        "-2.5" -> decimal { negative: "true" integer_part: "دو" fractional_part: "پنج دهم" }

    In Persian:
    - Decimal separator can be "." or "٫" (Arabic decimal separator)
    - Fractional parts are read as "X از Y" (X out of Y) or "X Yم" (X Yth)
    - e.g., 0.5 = "پنج دهم" (five tenths)
    - e.g., 0.25 = "بیست و پنج صدم" (twenty-five hundredths)

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="decimal", kind="classify", deterministic=deterministic)

        integer_part = cardinal.cardinal_numbers
        cardinal_numbers_with_leading_zeros = cardinal.cardinal_numbers_with_leading_zeros

        self.integer_part = pynini.closure(integer_part, 0, 1)

        # Decimal separators: "." or Arabic decimal separator "٫"
        self.separator = pynini.union(".", "٫", ",")

        # Fractional denominators in Persian
        # 1 digit after decimal -> دهم (tenths)
        # 2 digits -> صدم (hundredths)
        # 3 digits -> هزارم (thousandths)
        add_of = pynutil.insert(" ")

        graph_fractional_one = NEMO_DIGIT @ cardinal_numbers_with_leading_zeros + add_of + pynutil.insert("دهم")
        graph_fractional_two = (
            (NEMO_DIGIT + NEMO_DIGIT) @ cardinal_numbers_with_leading_zeros + add_of + pynutil.insert("صدم")
        )
        graph_fractional_three = (
            (NEMO_DIGIT + NEMO_DIGIT + NEMO_DIGIT) @ cardinal_numbers_with_leading_zeros
            + add_of
            + pynutil.insert("هزارم")
        )

        graph_fractional = graph_fractional_one | graph_fractional_two | graph_fractional_three
        self.graph_fractional = graph_fractional

        # Integer part with quotes
        graph_integer = pynutil.insert('integer_part: "') + self.integer_part + pynutil.insert('"')
        graph_integer_or_none = graph_integer | pynutil.insert('integer_part: "صفر"', weight=0.001)

        # Fractional part with quotes
        graph_fractional_final = (
            pynutil.insert('fractional_part: "')
            + pynutil.delete(self.separator)
            + graph_fractional
            + pynutil.insert('"')
        )

        # Handle negative numbers
        optional_minus = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1)

        self.final_graph_decimal = optional_minus + graph_integer_or_none + insert_space + graph_fractional_final

        final_graph = self.add_tokens(self.final_graph_decimal)
        self.fst = final_graph.optimize()
