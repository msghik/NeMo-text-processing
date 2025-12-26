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
    insert_space,
)


class FractionFst(GraphFst):
    """
    Finite state transducer for verbalizing fractions, e.g.
        fraction { numerator: "یک" denominator: "دوم" } -> "یک دوم"
        fraction { integer_part: "دو" numerator: "یک" denominator: "دوم" } -> "دو و یک دوم"

    Args:
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="fraction", kind="verbalize", deterministic=deterministic)

        # Optional integer part
        integer_part = (
            pynutil.delete('integer_part: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_integer = pynini.closure(
            integer_part + pynutil.insert(" و ") + pynutil.delete(" "),
            0,
            1,
        )

        # Numerator
        numerator = (
            pynutil.delete('numerator: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # Denominator
        denominator = (
            pynutil.delete('denominator: "')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        self.graph = optional_integer + numerator + insert_space + pynutil.delete(" ") + denominator

        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()
