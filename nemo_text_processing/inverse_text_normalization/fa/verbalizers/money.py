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


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money in ITN.
        e.g. money { integer_part: "100" currency: "$" } -> "$100"
        e.g. money { integer_part: "1000" currency: "تومان" } -> "1000 تومان"
    """

    def __init__(self):
        super().__init__(name="money", kind="verbalize")

        # Integer part
        integer_graph = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # Currency
        currency_graph = (
            pynutil.delete("currency:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # For symbol currencies ($, €, £, ¥) - symbol before amount
        symbol_currencies = pynini.union("$", "€", "£", "¥")
        currency_symbol = (
            pynutil.delete("currency:")
            + delete_space
            + pynutil.delete('"')
            + symbol_currencies
            + pynutil.delete('"')
        )

        # For word currencies (تومان, ریال) - amount before currency
        word_currencies = pynini.union("تومان", "ریال")
        currency_word = (
            pynutil.delete("currency:")
            + delete_space
            + pynutil.delete('"')
            + word_currencies
            + pynutil.delete('"')
        )

        # Symbol format: $100
        graph_symbol = currency_symbol + delete_space + integer_graph

        # Word format: 1000 تومان
        graph_word = integer_graph + pynutil.insert(" ") + delete_space + currency_word

        # Combine both formats
        graph = graph_symbol | graph_word

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
