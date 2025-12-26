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


class DateFst(GraphFst):
    """
    Finite state transducer for classifying dates in inverse text normalization.
    Converts spoken form to written form, e.g.
        "پانزدهم مرداد هزار و چهارصد و دو" -> date { day: "15" month: "5" year: "1402" }

    Args:
        tn_cardinal: cardinal FST from text normalization (to invert)
    """

    def __init__(self, tn_cardinal):
        super().__init__(name="date", kind="classify")

        # Invert cardinal for day and year
        cardinal_graph = pynini.invert(tn_cardinal.cardinal_numbers).optimize()

        # Month mapping (Persian month names -> numbers)
        month_jalali_map = pynini.string_map(
            [
                ("فروردین", "01"),
                ("اردیبهشت", "02"),
                ("خرداد", "03"),
                ("تیر", "04"),
                ("مرداد", "05"),
                ("شهریور", "06"),
                ("مهر", "07"),
                ("آبان", "08"),
                ("آذر", "09"),
                ("دی", "10"),
                ("بهمن", "11"),
                ("اسفند", "12"),
            ]
        )

        month_gregorian_map = pynini.string_map(
            [
                ("ژانویه", "01"),
                ("فوریه", "02"),
                ("مارس", "03"),
                ("آوریل", "04"),
                ("مه", "05"),
                ("ژوئن", "06"),
                ("ژوئیه", "07"),
                ("اوت", "08"),
                ("سپتامبر", "09"),
                ("اکتبر", "10"),
                ("نوامبر", "11"),
                ("دسامبر", "12"),
            ]
        )

        month_graph = month_jalali_map | month_gregorian_map

        # Day with ordinal suffix removal (م or ام)
        day_graph = cardinal_graph + pynutil.delete(pynini.union("م", "ام"))

        # Components
        graph_day = pynutil.insert('day: "') + day_graph + pynutil.insert('"')
        graph_month = pynutil.insert('month: "') + month_graph + pynutil.insert('"')
        graph_year = pynutil.insert('year: "') + cardinal_graph + pynutil.insert('"')

        # Persian date format: Day + Month + Year
        # e.g., "پانزدهم مرداد هزار و چهارصد و دو"
        final_graph = (
            graph_day
            + pynutil.delete(" ")
            + insert_space
            + graph_month
            + pynutil.delete(" ")
            + insert_space
            + graph_year
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
