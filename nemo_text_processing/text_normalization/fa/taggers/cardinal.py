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


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        "123" -> cardinal { integer: "صد و بیست و سه" }

    Persian numbers follow this pattern:
    - 0: صفر
    - 1-9: یک، دو، سه، چهار، پنج، شش، هفت، هشت، نه
    - 10-19: ده، یازده، دوازده، سیزده، چهارده، پانزده، شانزده، هفده، هجده، نوزده
    - 20, 30, ..., 90: بیست، سی، چهل، پنجاه، شصت، هفتاد، هشتاد، نود
    - Compound numbers use "و" (va = and): بیست و یک (21)

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="classify", deterministic=deterministic)

        # Load data files
        graph_zero = pynini.string_file(get_abs_path("data/number/zero.tsv"))
        graph_digit = pynini.string_file(get_abs_path("data/number/digit.tsv"))
        graph_teens = pynini.string_file(get_abs_path("data/number/teens.tsv"))
        graph_tens = pynini.string_file(get_abs_path("data/number/tens.tsv"))
        graph_hundreds = pynini.string_file(get_abs_path("data/number/hundreds.tsv"))

        # Insert Persian conjunction "و" (and)
        insert_va = pynutil.insert(" و ")

        # Single digits: 1-9
        graph_single_digit = graph_digit

        # Teens: 10-19 (special forms)
        graph_teen = pynini.cross("1", "") + graph_teens

        # Tens with zero: 20, 30, ..., 90
        graph_tens_zero = graph_tens + pynutil.delete("0")

        # Tens with digit: 21-29, 31-39, ..., 91-99
        # In Persian: بیست و یک (twenty and one)
        graph_tens_with_digit = graph_tens + insert_va + graph_digit

        # All two-digit numbers
        graph_two_digit = graph_teen | graph_tens_zero | graph_tens_with_digit

        # Hundreds: 100, 200, ..., 900
        graph_hundreds_zero = graph_hundreds + pynutil.delete("00")

        # Hundreds with tens: 110-199, 210-299, etc.
        graph_hundreds_with_two_digit = graph_hundreds + insert_va + graph_two_digit

        # Hundreds with single digit: 101-109, 201-209, etc.
        graph_hundreds_with_single = graph_hundreds + pynutil.delete("0") + insert_va + graph_digit

        # All three-digit numbers
        graph_three_digit = graph_hundreds_zero | graph_hundreds_with_two_digit | graph_hundreds_with_single

        # Thousands (basic support for 1000-9999)
        # هزار (thousand)
        graph_thousand_single = pynini.cross("1", "هزار") + pynutil.delete("000")
        graph_thousand_digit = graph_digit + pynutil.insert(" هزار") + pynutil.delete("000")

        graph_thousand_with_hundreds = graph_digit + pynutil.insert(" هزار") + insert_va + graph_three_digit
        graph_thousand_with_tens = (
            graph_digit + pynutil.insert(" هزار") + pynutil.delete("0") + insert_va + graph_two_digit
        )
        graph_thousand_with_digit = (
            graph_digit + pynutil.insert(" هزار") + pynutil.delete("00") + insert_va + graph_single_digit
        )

        graph_one_thousand_with_hundreds = pynini.cross("1", "هزار") + insert_va + graph_three_digit
        graph_one_thousand_with_tens = pynini.cross("1", "هزار") + pynutil.delete("0") + insert_va + graph_two_digit
        graph_one_thousand_with_digit = (
            pynini.cross("1", "هزار") + pynutil.delete("00") + insert_va + graph_single_digit
        )

        graph_four_digit = (
            graph_thousand_single
            | graph_thousand_digit
            | graph_thousand_with_hundreds
            | graph_thousand_with_tens
            | graph_thousand_with_digit
            | graph_one_thousand_with_hundreds
            | graph_one_thousand_with_tens
            | graph_one_thousand_with_digit
        )

        # Combine all cardinal graphs
        graph = graph_zero | graph_single_digit | graph_two_digit | graph_three_digit | graph_four_digit

        self.cardinal_numbers = graph.optimize()

        # Handle leading zeros
        leading_zeros = pynini.closure(pynini.cross("0", ""))
        self.cardinal_numbers_with_leading_zeros = (leading_zeros + self.cardinal_numbers).optimize()

        # Handle negative numbers
        self.optional_minus_graph = pynini.closure(pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1)

        # Final graph with tagging
        final_graph = (
            self.optional_minus_graph
            + pynutil.insert('integer: "')
            + self.cardinal_numbers_with_leading_zeros
            + pynutil.insert('"')
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
