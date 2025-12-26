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
    Finite state transducer for verbalizing decimals in ITN.
        e.g. decimal { integer_part: "3" fractional_part: "14" } -> "3.14"
        e.g. decimal { negative: "-" integer_part: "2" fractional_part: "5" } -> "-2.5"
    """

    def __init__(self):
        super().__init__(name="decimal", kind="verbalize")

        # Negative sign
        negative_graph = (
            pynutil.delete("negative:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_negative = pynini.closure(negative_graph + delete_space, 0, 1)

        # Integer part
        integer_graph = (
            pynutil.delete("integer_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # Fractional part
        fractional_graph = (
            pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # Combine: negative + integer + "." + fractional
        graph = optional_negative + integer_graph + pynutil.insert(".") + delete_space + fractional_graph

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
