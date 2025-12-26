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
    NEMO_DIGIT,
    GraphFst,
    insert_space,
)
from nemo_text_processing.text_normalization.fa.utils import get_abs_path


class DateFst(GraphFst):
    """
    Finite state transducer for classifying dates, e.g.
        "1402/05/15" -> date { year: "هزار و چهارصد و دو" month: "مرداد" day: "پانزده" }
        "15/05/1402" -> date { day: "پانزده" month: "مرداد" year: "هزار و چهارصد و دو" }
        "2023-12-25" -> date { year: "دو هزار و بیست و سه" month: "دسامبر" day: "بیست و پنج" }

    Supports:
    - Jalali (Persian/Solar Hijri) calendar: YYYY/MM/DD
    - Gregorian calendar: YYYY-MM-DD, DD/MM/YYYY

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="date", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.cardinal_numbers

        # Month names
        months_jalali = pynini.string_file(get_abs_path("data/date/months_jalali.tsv"))
        months_gregorian = pynini.string_file(get_abs_path("data/date/months_gregorian.tsv"))

        # Day: 1-31
        delete_leading_zero = (NEMO_DIGIT + NEMO_DIGIT) | (
            pynini.closure(pynutil.delete("0"), 0, 1) + NEMO_DIGIT
        )
        labels_day = [str(x) for x in range(1, 32)]
        graph_day = delete_leading_zero @ pynini.union(*labels_day) @ cardinal_graph

        # Month: 1-12 (convert to name)
        labels_month = [str(x) for x in range(1, 13)]
        graph_month_num = delete_leading_zero @ pynini.union(*labels_month)
        graph_month_jalali = graph_month_num @ months_jalali
        graph_month_gregorian = graph_month_num @ months_gregorian

        # Year (4 digits for Jalali ~1300-1500, or Gregorian ~1900-2100)
        graph_year = pynini.closure(NEMO_DIGIT, 4, 4) @ cardinal_graph

        # Build components
        final_day = pynutil.insert('day: "') + graph_day + pynutil.insert('"')
        final_month_jalali = pynutil.insert('month: "') + graph_month_jalali + pynutil.insert('"')
        final_month_gregorian = pynutil.insert('month: "') + graph_month_gregorian + pynutil.insert('"')
        final_year = pynutil.insert('year: "') + graph_year + pynutil.insert('"')

        # Jalali format: YYYY/MM/DD (most common in Persian)
        graph_jalali_ymd = (
            final_year
            + pynutil.delete("/")
            + insert_space
            + final_month_jalali
            + pynutil.delete("/")
            + insert_space
            + final_day
        )

        # Alternative Jalali: DD/MM/YYYY
        graph_jalali_dmy = (
            final_day
            + pynutil.delete("/")
            + insert_space
            + final_month_jalali
            + pynutil.delete("/")
            + insert_space
            + final_year
        )

        # Gregorian format: YYYY-MM-DD
        graph_gregorian_ymd = (
            final_year
            + pynutil.delete("-")
            + insert_space
            + final_month_gregorian
            + pynutil.delete("-")
            + insert_space
            + final_day
        )

        # Gregorian: DD-MM-YYYY
        graph_gregorian_dmy = (
            final_day
            + pynutil.delete("-")
            + insert_space
            + final_month_gregorian
            + pynutil.delete("-")
            + insert_space
            + final_year
        )

        self.graph = (
            graph_jalali_ymd
            | pynutil.add_weight(graph_jalali_dmy, 0.1)
            | graph_gregorian_ymd
            | pynutil.add_weight(graph_gregorian_dmy, 0.1)
        )

        final_graph = self.add_tokens(self.graph)
        self.fst = final_graph.optimize()
