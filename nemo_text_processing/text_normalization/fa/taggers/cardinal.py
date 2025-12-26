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

from nemo_text_processing.text_normalization.fa.graph_utils import GraphFst
from nemo_text_processing.text_normalization.fa.utils import get_abs_path


class CardinalFst(GraphFst):
    """
    Finite state transducer for classifying cardinals, e.g.
        "123" -> cardinal { integer: "صد و بیست و سه" }
        "1000" -> cardinal { integer: "هزار" }
        "1000000" -> cardinal { integer: "یک میلیون" }

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

        # All two-digit numbers (10-99)
        graph_two_digit = graph_teen | graph_tens_zero | graph_tens_with_digit

        # Hundreds: 100, 200, ..., 900
        graph_hundreds_zero = graph_hundreds + pynutil.delete("00")

        # Hundreds with tens: 110-199, 210-299, etc.
        graph_hundreds_with_two_digit = graph_hundreds + insert_va + graph_two_digit

        # Hundreds with single digit: 101-109, 201-209, etc.
        graph_hundreds_with_single = (
            graph_hundreds + pynutil.delete("0") + insert_va + graph_digit
        )

        # All three-digit numbers (100-999)
        graph_three_digit = (
            graph_hundreds_zero
            | graph_hundreds_with_two_digit
            | graph_hundreds_with_single
        )

        # ===== THOUSANDS (1,000 - 999,999) =====
        # هزار (thousand)
        # "1000" -> "هزار" (one thousand - the "one" is implicit)
        graph_thousand_single = pynini.cross("1", "هزار") + pynutil.delete("000")

        # "2000"-"9000" -> "دو هزار", "سه هزار", etc.
        graph_thousand_digit = (
            graph_digit + pynutil.insert(" هزار") + pynutil.delete("000")
        )

        # "1001"-"1999" -> "هزار و ..."
        graph_one_thousand_with_hundreds = (
            pynini.cross("1", "هزار") + insert_va + graph_three_digit
        )
        graph_one_thousand_with_tens = (
            pynini.cross("1", "هزار")
            + pynutil.delete("0")
            + insert_va
            + graph_two_digit
        )
        graph_one_thousand_with_digit = (
            pynini.cross("1", "هزار")
            + pynutil.delete("00")
            + insert_va
            + graph_single_digit
        )

        # "2001"-"9999" -> "X هزار و ..."
        graph_thousand_with_hundreds = (
            graph_digit + pynutil.insert(" هزار") + insert_va + graph_three_digit
        )
        graph_thousand_with_tens = (
            graph_digit
            + pynutil.insert(" هزار")
            + pynutil.delete("0")
            + insert_va
            + graph_two_digit
        )
        graph_thousand_with_digit = (
            graph_digit
            + pynutil.insert(" هزار")
            + pynutil.delete("00")
            + insert_va
            + graph_single_digit
        )

        # 4-digit numbers (1000-9999)
        graph_four_digit = (
            graph_thousand_single
            | graph_thousand_digit
            | graph_one_thousand_with_hundreds
            | graph_one_thousand_with_tens
            | graph_one_thousand_with_digit
            | graph_thousand_with_hundreds
            | graph_thousand_with_tens
            | graph_thousand_with_digit
        )

        # 5-digit numbers (10,000 - 99,999)
        # "10000" -> "ده هزار"
        graph_ten_thousands_zero = (
            graph_two_digit + pynutil.insert(" هزار") + pynutil.delete("000")
        )
        graph_ten_thousands_with_hundreds = (
            graph_two_digit + pynutil.insert(" هزار") + insert_va + graph_three_digit
        )
        graph_ten_thousands_with_tens = (
            graph_two_digit
            + pynutil.insert(" هزار")
            + pynutil.delete("0")
            + insert_va
            + graph_two_digit
        )
        graph_ten_thousands_with_digit = (
            graph_two_digit
            + pynutil.insert(" هزار")
            + pynutil.delete("00")
            + insert_va
            + graph_single_digit
        )

        graph_five_digit = (
            graph_ten_thousands_zero
            | graph_ten_thousands_with_hundreds
            | graph_ten_thousands_with_tens
            | graph_ten_thousands_with_digit
        )

        # 6-digit numbers (100,000 - 999,999)
        # "100000" -> "صد هزار"
        graph_hundred_thousands_zero = (
            graph_three_digit + pynutil.insert(" هزار") + pynutil.delete("000")
        )
        graph_hundred_thousands_with_hundreds = (
            graph_three_digit + pynutil.insert(" هزار") + insert_va + graph_three_digit
        )
        graph_hundred_thousands_with_tens = (
            graph_three_digit
            + pynutil.insert(" هزار")
            + pynutil.delete("0")
            + insert_va
            + graph_two_digit
        )
        graph_hundred_thousands_with_digit = (
            graph_three_digit
            + pynutil.insert(" هزار")
            + pynutil.delete("00")
            + insert_va
            + graph_single_digit
        )

        graph_six_digit = (
            graph_hundred_thousands_zero
            | graph_hundred_thousands_with_hundreds
            | graph_hundred_thousands_with_tens
            | graph_hundred_thousands_with_digit
        )

        # ===== MILLIONS (1,000,000 - 999,999,999) =====
        # میلیون (million)
        # Unlike "thousand", "million" requires "یک" (one) explicitly
        # "1000000" -> "یک میلیون"
        graph_million_single = pynini.cross("1", "یک میلیون") + pynutil.delete("000000")
        graph_million_digit = (
            graph_digit + pynutil.insert(" میلیون") + pynutil.delete("000000")
        )

        # Millions with remainder (1-999)
        graph_one_million_with_hundreds = (
            pynini.cross("1", "یک میلیون")
            + pynutil.delete("000")
            + insert_va
            + graph_three_digit
        )
        graph_one_million_with_tens = (
            pynini.cross("1", "یک میلیون")
            + pynutil.delete("0000")
            + insert_va
            + graph_two_digit
        )
        graph_one_million_with_digit = (
            pynini.cross("1", "یک میلیون")
            + pynutil.delete("00000")
            + insert_va
            + graph_single_digit
        )

        graph_million_with_hundreds = (
            graph_digit
            + pynutil.insert(" میلیون")
            + pynutil.delete("000")
            + insert_va
            + graph_three_digit
        )
        graph_million_with_tens = (
            graph_digit
            + pynutil.insert(" میلیون")
            + pynutil.delete("0000")
            + insert_va
            + graph_two_digit
        )
        graph_million_with_digit = (
            graph_digit
            + pynutil.insert(" میلیون")
            + pynutil.delete("00000")
            + insert_va
            + graph_single_digit
        )

        # Millions with thousands (1,000 - 999,999)
        # Build graph for thousands component (1-999999 with leading zeros removed)
        graph_thousands_component = (
            graph_four_digit
            | (pynutil.delete("0") + graph_three_digit)
            | (pynutil.delete("00") + graph_two_digit)
            | (pynutil.delete("000") + graph_single_digit)
        )

        graph_one_million_with_thousands = (
            pynini.cross("1", "یک میلیون") + insert_va + graph_thousands_component
        )
        graph_million_with_thousands = (
            graph_digit
            + pynutil.insert(" میلیون")
            + insert_va
            + graph_thousands_component
        )

        graph_seven_digit = (
            graph_million_single
            | graph_million_digit
            | graph_one_million_with_hundreds
            | graph_one_million_with_tens
            | graph_one_million_with_digit
            | graph_million_with_hundreds
            | graph_million_with_tens
            | graph_million_with_digit
            | graph_one_million_with_thousands
            | graph_million_with_thousands
        )

        # 8-digit numbers (10,000,000 - 99,999,999)
        graph_ten_million_zero = (
            graph_two_digit + pynutil.insert(" میلیون") + pynutil.delete("000000")
        )
        graph_ten_million_with_thousands = (
            graph_two_digit
            + pynutil.insert(" میلیون")
            + insert_va
            + graph_thousands_component
        )

        graph_eight_digit = graph_ten_million_zero | graph_ten_million_with_thousands

        # 9-digit numbers (100,000,000 - 999,999,999)
        graph_hundred_million_zero = (
            graph_three_digit + pynutil.insert(" میلیون") + pynutil.delete("000000")
        )
        graph_hundred_million_with_thousands = (
            graph_three_digit
            + pynutil.insert(" میلیون")
            + insert_va
            + graph_thousands_component
        )

        graph_nine_digit = (
            graph_hundred_million_zero | graph_hundred_million_with_thousands
        )

        # ===== BILLIONS (1,000,000,000 - 999,999,999,999) =====
        # میلیارد (billion)
        graph_billion_single = pynini.cross("1", "یک میلیارد") + pynutil.delete(
            "000000000"
        )
        graph_billion_digit = (
            graph_digit + pynutil.insert(" میلیارد") + pynutil.delete("000000000")
        )

        # Billions with millions component
        graph_millions_component = (
            graph_seven_digit
            | (pynutil.delete("0") + graph_six_digit)
            | (pynutil.delete("00") + graph_five_digit)
            | (pynutil.delete("000") + graph_four_digit)
            | (pynutil.delete("0000") + graph_three_digit)
            | (pynutil.delete("00000") + graph_two_digit)
            | (pynutil.delete("000000") + graph_single_digit)
        )

        graph_one_billion_with_millions = (
            pynini.cross("1", "یک میلیارد") + insert_va + graph_millions_component
        )
        graph_billion_with_millions = (
            graph_digit
            + pynutil.insert(" میلیارد")
            + insert_va
            + graph_millions_component
        )

        graph_ten_digit = (
            graph_billion_single
            | graph_billion_digit
            | graph_one_billion_with_millions
            | graph_billion_with_millions
        )

        # Combine all cardinal graphs
        graph = (
            graph_zero
            | graph_single_digit
            | graph_two_digit
            | graph_three_digit
            | graph_four_digit
            | graph_five_digit
            | graph_six_digit
            | graph_seven_digit
            | graph_eight_digit
            | graph_nine_digit
            | graph_ten_digit
        )

        self.cardinal_numbers = graph.optimize()

        # Handle leading zeros
        leading_zeros = pynini.closure(pynini.cross("0", ""))
        self.cardinal_numbers_with_leading_zeros = (
            leading_zeros + self.cardinal_numbers
        ).optimize()

        # Handle negative numbers
        self.optional_minus_graph = pynini.closure(
            pynutil.insert("negative: ") + pynini.cross("-", '"true" '), 0, 1
        )

        # Final graph with tagging
        final_graph = (
            self.optional_minus_graph
            + pynutil.insert('integer: "')
            + self.cardinal_numbers_with_leading_zeros
            + pynutil.insert('"')
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
