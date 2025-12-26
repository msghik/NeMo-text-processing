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

from nemo_text_processing.text_normalization.fa.graph_utils import NEMO_DIGIT, GraphFst
from nemo_text_processing.text_normalization.fa.utils import get_abs_path


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinals, e.g.
        "1st" -> ordinal { integer: "اول" }
        "2nd" -> ordinal { integer: "دوم" }
        "3rd" -> ordinal { integer: "سوم" }
        "1م" -> ordinal { integer: "اول" }
        "2م" -> ordinal { integer: "دوم" }

    Persian ordinals:
    - اول (avval) - first
    - دوم (dovvom) - second
    - سوم (sevvom) - third
    - چهارم (chaharom) - fourth
    - etc.

    Common suffixes: ام, م (Persian), st, nd, rd, th (English)

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="ordinal", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.cardinal_numbers
        graph_ordinal = pynini.string_file(get_abs_path("data/number/ordinal.tsv"))

        # Persian ordinal suffixes: ام or م
        persian_suffix = pynutil.delete(pynini.union("ام", "م"))

        # English ordinal suffixes (for mixed usage)
        english_suffix = pynutil.delete(pynini.union("st", "nd", "rd", "th", "ST", "ND", "RD", "TH"))

        # Numbers 1-20 have direct ordinal mapping
        direct_ordinals = pynini.string_map([
            ("1", "اول"),
            ("2", "دوم"),
            ("3", "سوم"),
            ("4", "چهارم"),
            ("5", "پنجم"),
            ("6", "ششم"),
            ("7", "هفتم"),
            ("8", "هشتم"),
            ("9", "نهم"),
            ("10", "دهم"),
            ("11", "یازدهم"),
            ("12", "دوازدهم"),
            ("13", "سیزدهم"),
            ("14", "چهاردهم"),
            ("15", "پانزدهم"),
            ("16", "شانزدهم"),
            ("17", "هفدهم"),
            ("18", "هجدهم"),
            ("19", "نوزدهم"),
            ("20", "بیستم"),
        ])

        # For numbers > 20, append ام to cardinal
        # e.g., 21 -> بیست و یک + م = بیست و یکم
        cardinal_with_suffix = cardinal_graph + pynutil.insert("م")

        # Graph with Persian suffix
        graph_persian = (
            pynini.closure(NEMO_DIGIT, 1, 2) @ direct_ordinals + persian_suffix
            | pynini.closure(NEMO_DIGIT, 1) @ cardinal_with_suffix + persian_suffix
        )

        # Graph with English suffix
        graph_english = (
            pynini.closure(NEMO_DIGIT, 1, 2) @ direct_ordinals + english_suffix
            | pynini.closure(NEMO_DIGIT, 1) @ cardinal_with_suffix + english_suffix
        )

        self.graph = graph_persian | graph_english

        final_graph = pynutil.insert('integer: "') + self.graph + pynutil.insert('"')
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
