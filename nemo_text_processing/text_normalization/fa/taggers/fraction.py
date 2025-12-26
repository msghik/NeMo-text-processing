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
from nemo_text_processing.text_normalization.fa.utils import get_abs_path


class FractionFst(GraphFst):
    """
    Finite state transducer for classifying fractions, e.g.
        "1/2" -> fraction { numerator: "یک" denominator: "دوم" }
        "3/4" -> fraction { numerator: "سه" denominator: "چهارم" }
        "2 1/2" -> fraction { integer_part: "دو" numerator: "یک" denominator: "دوم" }

    Persian fractions:
    - نیم (nim) - half (1/2)
    - یک سوم (yek sevvom) - one third
    - دو سوم (do sevvom) - two thirds
    - یک چهارم (yek chaharom) - one quarter

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="fraction", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.cardinal_numbers
        graph_denominator = pynini.string_file(get_abs_path("data/number/denominator.tsv"))

        # Numerator
        graph_numerator = pynutil.insert('numerator: "') + cardinal_graph + pynutil.insert('"')

        # Denominator (ordinal form for small numbers)
        graph_denom_small = (
            pynutil.insert('denominator: "')
            + graph_denominator
            + pynutil.insert('"')
        )

        # For larger denominators, use cardinal + م
        graph_denom_large = (
            pynutil.insert('denominator: "')
            + cardinal_graph
            + pynutil.insert('م"')
        )

        # Fraction separator
        divider = pynini.cross("/", '" ') | pynini.cross(" / ", '" ')

        # Small denominators (2-10)
        small_denom = pynini.union("2", "3", "4", "5", "6", "7", "8", "9", "10")
        graph_fraction_small = graph_numerator + divider + (small_denom @ graph_denom_small)

        # Larger denominators
        graph_fraction_large = graph_numerator + divider + graph_denom_large

        # Combined fraction graph
        fraction_graph = graph_fraction_small | pynutil.add_weight(graph_fraction_large, 0.1)

        # Optional integer part (e.g., "2 1/2")
        integer_part = (
            pynutil.insert('integer_part: "')
            + cardinal_graph
            + pynutil.insert('" ')
        )
        optional_integer = pynini.closure(integer_part + pynutil.delete(" "), 0, 1)

        # Special case: 1/2 = نیم
        half = pynini.cross("1/2", 'numerator: "یک" denominator: "دوم"')
        quarter = pynini.cross("1/4", 'numerator: "یک" denominator: "چهارم"')
        three_quarters = pynini.cross("3/4", 'numerator: "سه" denominator: "چهارم"')

        special_fractions = half | quarter | three_quarters

        self.graph = optional_integer + (special_fractions | fraction_graph)

        final_graph = self.add_tokens(self.graph)
        self.fst = final_graph.optimize()
