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

from nemo_text_processing.text_normalization.fa.graph_utils import GraphFst, insert_space


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money in inverse text normalization.
    Converts spoken form to written form, e.g.
        "صد دلار" -> money { integer_part: "100" currency: "$" }
        "هزار تومان" -> money { integer_part: "1000" currency: "تومان" }
        "پنجاه یورو" -> money { integer_part: "50" currency: "€" }

    Args:
        itn_cardinal_tagger: cardinal ITN tagger
    """

    def __init__(self, itn_cardinal_tagger):
        super().__init__(name="money", kind="classify")

        cardinal_graph = itn_cardinal_tagger.graph

        # Currency mapping (spoken -> written symbol)
        currency_map = pynini.string_map([
            ("دلار", "$"),
            ("یورو", "€"),
            ("پوند", "£"),
            ("ین", "¥"),
            ("ریال", "ریال"),
            ("تومان", "تومان"),
        ])

        # Integer part
        graph_integer = pynutil.insert('integer_part: "') + cardinal_graph + pynutil.insert('"')

        # Currency
        graph_currency = pynutil.insert('currency: "') + currency_map + pynutil.insert('"')

        # Money: amount + currency
        # e.g., "صد دلار" -> integer_part: "100" currency: "$"
        final_graph = graph_integer + pynutil.delete(" ") + insert_space + graph_currency

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
