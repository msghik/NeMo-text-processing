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


class MoneyFst(GraphFst):
    """
    Finite state transducer for verbalizing money, e.g.
        money { integer_part: "صد" currency: "دلار" } -> "صد دلار"
        money { integer_part: "هزار" currency: "تومان" } -> "هزار تومان"

    Args:
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)

        # Integer part
        integer_part = (
            pynutil.delete('integer_part: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # Fractional part (for decimal money)
        fractional_part = (
            pynutil.delete('fractional_part: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # Currency
        currency = (
            pynutil.delete('currency: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # Graph: Amount + Currency
        # Handle different orderings in the tagged input
        graph_amount_currency = integer_part + delete_space + insert_space + currency
        graph_currency_amount = currency + delete_space + insert_space + integer_part

        # With fractional part
        graph_with_fractional = (
            integer_part
            + pynutil.insert(" و ")
            + delete_space
            + fractional_part
            + delete_space
            + insert_space
            + currency
        )

        self.graph = graph_amount_currency | graph_currency_amount | pynutil.add_weight(graph_with_fractional, 0.1)

        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()
