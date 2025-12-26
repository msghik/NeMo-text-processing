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
    NEMO_DIGIT,
    NEMO_SIGMA,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.fa.utils import get_abs_path


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money expressions, e.g.
        "$100" -> money { integer_part: "صد" currency: "دلار" }
        "50€" -> money { integer_part: "پنجاه" currency: "یورو" }
        "1000 تومان" -> money { integer_part: "هزار" currency: "تومان" }
        "500,000 ریال" -> money { integer_part: "پانصد هزار" currency: "ریال" }

    Persian currency:
    - ریال (Rial) - official currency
    - تومان (Toman) - commonly used (1 Toman = 10 Rial)

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.cardinal_numbers_with_leading_zeros
        currency_graph = pynini.string_file(get_abs_path("data/money/currency.tsv"))

        # Build currency component
        graph_currency = pynutil.insert('currency: "') + currency_graph + pynutil.insert('"')

        # Build integer part component
        graph_integer = pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')

        # Handle thousand separators (comma or space)
        # e.g., 1,000,000 or 1 000 000
        delete_separator = pynutil.delete(pynini.union(",", " ", "،"))  # Include Persian comma
        number_with_separator = pynini.closure(NEMO_DIGIT, 1) + pynini.closure(
            delete_separator + pynini.closure(NEMO_DIGIT, 3, 3)
        )
        graph_integer_with_sep = (
            pynutil.insert('integer_part: "')
            + (number_with_separator @ cardinal_graph)
            + pynutil.insert('"')
        )

        # Currency symbol before number: $100, €50, £20
        graph_symbol_before = (
            graph_currency
            + insert_space
            + (graph_integer | graph_integer_with_sep)
        )

        # Currency symbol/word after number: 100$, 50€, 1000 تومان
        graph_symbol_after = (
            (graph_integer | graph_integer_with_sep)
            + pynini.closure(pynutil.delete(" "), 0, 1)
            + insert_space
            + graph_currency
        )

        # Decimal money (e.g., $10.50) - simplified
        graph_decimal_money = (
            graph_currency
            + insert_space
            + pynutil.insert('integer_part: "')
            + cardinal_graph
            + pynutil.insert('"')
            + pynutil.delete(".")
            + insert_space
            + pynutil.insert('fractional_part: "')
            + (pynini.closure(NEMO_DIGIT, 1, 2) @ cardinal_graph)
            + pynutil.insert('"')
        )

        self.graph = graph_symbol_before | graph_symbol_after | pynutil.add_weight(graph_decimal_money, 0.1)

        final_graph = self.add_tokens(self.graph)
        self.fst = final_graph.optimize()
