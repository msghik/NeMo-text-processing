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

from nemo_text_processing.text_normalization.fa.graph_utils import NEMO_DIGIT, GraphFst, delete_space, insert_space
from nemo_text_processing.text_normalization.fa.utils import get_abs_path


class TimeFst(GraphFst):
    """
    Finite state transducer for classifying time expressions, e.g.
        "3:30" -> time { hours: "سه" minutes: "سی" }
        "12:00" -> time { hours: "دوازده" }
        "3:30 pm" -> time { hours: "سه" minutes: "سی" suffix: "بعد از ظهر" }
        "15:45" -> time { hours: "پانزده" minutes: "چهل و پنج" }

    Persian time expressions:
    - ساعت سه و نیم (saat seh va nim) - half past three
    - ساعت چهار و ربع (saat chahar va rob) - quarter past four
    - ساعت پنج (saat panj) - five o'clock

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="time", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.cardinal_numbers

        # Load time suffix (am/pm)
        suffix_graph = pynini.string_file(get_abs_path("data/time/suffix.tsv"))

        # Hours: 0-23
        labels_hour = [str(x) for x in range(0, 24)]

        # Delete leading zero for hours
        delete_leading_zero = (NEMO_DIGIT + NEMO_DIGIT) | (pynini.closure(pynutil.delete("0"), 0, 1) + NEMO_DIGIT)

        graph_hour = delete_leading_zero @ pynini.union(*labels_hour) @ cardinal_graph

        # Minutes: 00-59
        labels_minute_single = [str(x) for x in range(1, 10)]
        labels_minute_double = [str(x) for x in range(10, 60)]

        graph_minute_single = pynini.union(*labels_minute_single) @ cardinal_graph
        graph_minute_double = pynini.union(*labels_minute_double) @ cardinal_graph

        # Special minutes
        graph_minute_30 = pynini.cross("30", "نیم")  # half
        graph_minute_15 = pynini.cross("15", "ربع")  # quarter
        graph_minute_45 = pynini.cross("45", "چهل و پنج")

        # Build hour graph
        final_graph_hour = pynutil.insert('hours: "') + graph_hour + pynutil.insert('"')

        # Build minute graph
        graph_minute = (
            pynini.cross("0", "") + graph_minute_single
            | graph_minute_double
            | graph_minute_30
            | graph_minute_15
            | graph_minute_45
        )
        final_graph_minute = pynutil.insert('minutes: "') + graph_minute + pynutil.insert('"')

        # Suffix (am/pm)
        final_suffix = pynutil.insert('suffix: "') + suffix_graph + pynutil.insert('"')
        final_suffix_optional = pynini.closure(delete_space + insert_space + final_suffix, 0, 1)

        # Time formats:
        # HH:MM, H:MM
        graph_hm = (
            final_graph_hour
            + pynutil.delete(":")
            + (pynutil.delete("00") | insert_space + final_graph_minute)
            + final_suffix_optional
        )

        # HH.MM format
        graph_hm_dot = (
            final_graph_hour
            + pynutil.delete(".")
            + (pynutil.delete("00") | insert_space + final_graph_minute)
            + final_suffix_optional
        )

        # H am/pm format (e.g., "3 pm")
        graph_h_suffix = final_graph_hour + delete_space + insert_space + final_suffix

        self.graph = graph_hm | graph_hm_dot | graph_h_suffix

        final_graph = self.add_tokens(self.graph)
        self.fst = final_graph.optimize()
