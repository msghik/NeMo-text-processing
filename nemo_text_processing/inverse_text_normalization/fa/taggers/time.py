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
from nemo_text_processing.inverse_text_normalization.fa.utils import get_abs_path


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time in inverse text normalization.
    Converts spoken form to written form, e.g.
        "ساعت سه و سی دقیقه" -> time { hours: "3" minutes: "30" }
        "ساعت دوازده" -> time { hours: "12" }
        "ساعت پنج بعد از ظهر" -> time { hours: "5" suffix: "pm" }

    Args:
        tn_cardinal: cardinal FST from text normalization (to invert)
    """

    def __init__(self, tn_cardinal):
        super().__init__(name="time", kind="classify")

        # Invert cardinal for hours and minutes
        cardinal_graph = pynini.invert(tn_cardinal.cardinal_numbers).optimize()

        # Special minute mappings
        minute_special = pynini.string_map([
            ("نیم", "30"),
            ("ربع", "15"),
        ])
        minute_graph = minute_special | cardinal_graph

        # Time prefix: "ساعت"
        time_prefix = pynutil.delete("ساعت ")

        # Hours
        graph_hours = pynutil.insert('hours: "') + cardinal_graph + pynutil.insert('"')

        # Minutes (optional)
        graph_minutes = (
            pynutil.delete(" و ")
            + insert_space
            + pynutil.insert('minutes: "')
            + minute_graph
            + pynutil.delete(" دقیقه")
            + pynutil.insert('"')
        )
        optional_minutes = pynini.closure(graph_minutes, 0, 1)

        # Suffix (am/pm) - optional
        suffix_map = pynini.string_map([
            ("صبح", "am"),
            ("بعد از ظهر", "pm"),
        ])
        graph_suffix = (
            pynutil.delete(" ")
            + insert_space
            + pynutil.insert('suffix: "')
            + suffix_map
            + pynutil.insert('"')
        )
        optional_suffix = pynini.closure(graph_suffix, 0, 1)

        final_graph = time_prefix + graph_hours + optional_minutes + optional_suffix

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
