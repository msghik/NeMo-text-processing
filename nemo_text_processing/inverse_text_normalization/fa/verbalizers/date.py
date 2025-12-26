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


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing dates in ITN.
        e.g. date { day: "15" month: "05" year: "1402" } -> "15/05/1402"
        e.g. date { day: "25" month: "12" year: "2023" } -> "25/12/2023"
    """

    def __init__(self):
        super().__init__(name="date", kind="verbalize")

        # Day
        day_graph = (
            pynutil.delete("day:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # Month
        month_graph = (
            pynutil.delete("month:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # Year
        year_graph = (
            pynutil.delete("year:")
            + delete_space
            + pynutil.delete('"')
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete('"')
        )

        # Combine: day + "/" + month + "/" + year
        graph = (
            day_graph
            + pynutil.insert("/")
            + delete_space
            + month_graph
            + pynutil.insert("/")
            + delete_space
            + year_graph
        )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
