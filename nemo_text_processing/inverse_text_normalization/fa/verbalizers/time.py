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


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time in ITN.
        e.g. time { hours: "3" minutes: "30" } -> "3:30"
        e.g. time { hours: "12" } -> "12:00"
        e.g. time { hours: "5" suffix: "pm" } -> "5:00 pm"
    """

    def __init__(self):
        super().__init__(name="time", kind="verbalize")

        # Hours
        hours_graph = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # Minutes (optional)
        minutes_graph = (
            pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_minutes = (
            pynutil.insert(":") + delete_space + minutes_graph
        ) | pynutil.insert(":00")

        # Suffix (optional)
        suffix_graph = (
            pynutil.delete("suffix:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )
        optional_suffix = pynini.closure(pynutil.insert(" ") + delete_space + suffix_graph, 0, 1)

        # Combine: hours + ":" + minutes + suffix
        graph = hours_graph + optional_minutes + optional_suffix

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
