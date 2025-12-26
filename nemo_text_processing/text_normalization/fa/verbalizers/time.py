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

from nemo_text_processing.text_normalization.fa.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space


class TimeFst(GraphFst):
    """
    Finite state transducer for verbalizing time, e.g.
        time { hours: "سه" minutes: "سی" } -> "ساعت سه و سی دقیقه"
        time { hours: "دوازده" } -> "ساعت دوازده"
        time { hours: "سه" minutes: "سی" suffix: "بعد از ظهر" } -> "ساعت سه و سی دقیقه بعد از ظهر"

    Args:
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="time", kind="verbalize", deterministic=deterministic)

        # Hours
        hours = pynutil.delete('hours: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        # Minutes
        minutes = pynutil.delete('minutes: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        # Suffix (am/pm)
        suffix = pynutil.delete('suffix: "') + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete('"')

        # Persian time prefix: ساعت (saat = o'clock)
        time_prefix = pynutil.insert("ساعت ")

        # Build graph: "ساعت X و Y دقیقه"
        # Hours only
        graph_hours_only = time_prefix + hours

        # Hours and minutes
        # Special case for نیم (half) and ربع (quarter)
        graph_hours_minutes = (
            time_prefix + hours + pynutil.insert(" و ") + delete_space + minutes + pynutil.insert(" دقیقه")
        )

        # With suffix
        optional_suffix = pynini.closure(delete_space + insert_space + suffix, 0, 1)

        self.graph = (graph_hours_minutes | graph_hours_only) + optional_suffix

        delete_tokens = self.delete_tokens(self.graph)
        self.fst = delete_tokens.optimize()
