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


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimals, e.g.
        decimal { integer_part: "سه" fractional_part: "چهارده صدم" } -> "سه ممیز چهارده صدم"

    In Persian, decimal point is verbalized as "ممیز" (momayez)

    Args:
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="decimal", kind="verbalize", deterministic=deterministic)

        # Handle optional negative
        optional_sign = pynini.closure(
            pynutil.delete("negative: ")
            + pynutil.delete('"')
            + pynini.cross("true", "منفی ")
            + pynutil.delete('"')
            + delete_space,
            0,
            1,
        )

        # Integer part
        integer_part = pynutil.delete('integer_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        # Fractional part
        fractional_part = (
            pynutil.delete('fractional_part: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')
        )

        # Persian decimal point word: ممیز (momayez)
        decimal_point = pynutil.insert(" ممیز ")

        self.graph = optional_sign + integer_part + decimal_point + delete_space + fractional_part

        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()
