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

from nemo_text_processing.text_normalization.fa.graph_utils import NEMO_ALPHA, NEMO_DIGIT, GraphFst, insert_space
from nemo_text_processing.text_normalization.fa.utils import get_abs_path


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measurements, e.g.
        "50%" -> measure { cardinal { integer: "پنجاه" } units: "درصد" }
        "100kg" -> measure { cardinal { integer: "صد" } units: "کیلوگرم" }
        "25°C" -> measure { cardinal { integer: "بیست و پنج" } units: "درجه سانتی‌گراد" }
        "10 کیلومتر" -> measure { cardinal { integer: "ده" } units: "کیلومتر" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst = None, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        cardinal_graph = cardinal.cardinal_numbers
        unit_graph = pynini.string_file(get_abs_path("data/measure/measurements.tsv"))

        # Build unit component
        graph_unit = pynutil.insert('units: "') + unit_graph + pynutil.insert('"')

        # Build cardinal component
        graph_cardinal = pynutil.insert('cardinal { integer: "') + cardinal_graph + pynutil.insert('" }')

        # Handle negative numbers
        optional_negative = pynini.closure(pynini.cross("-", "منفی "), 0, 1)
        graph_cardinal_with_neg = (
            pynutil.insert('cardinal { integer: "') + optional_negative + cardinal_graph + pynutil.insert('" }')
        )

        # Number + Unit (e.g., "50kg", "100 km", "25°C")
        graph_number_unit = (
            graph_cardinal_with_neg + pynini.closure(pynutil.delete(" "), 0, 1) + insert_space + graph_unit
        )

        # Unit + Number (less common, e.g., "kg 50")
        graph_unit_number = (
            graph_unit + pynini.closure(pynutil.delete(" "), 0, 1) + insert_space + graph_cardinal_with_neg
        )

        # Special: percentage
        graph_percentage = (
            pynutil.insert('cardinal { integer: "')
            + cardinal_graph
            + pynutil.insert('" }')
            + pynutil.delete("%")
            + pynutil.insert(' units: "درصد"')
        )

        # Handle decimal measurements if decimal FST is provided
        if decimal is not None:
            graph_decimal = pynutil.insert("decimal { ") + decimal.final_graph_decimal + pynutil.insert(" }")
            graph_decimal_unit = graph_decimal + pynini.closure(pynutil.delete(" "), 0, 1) + insert_space + graph_unit
            self.graph = (
                graph_number_unit
                | graph_percentage
                | pynutil.add_weight(graph_unit_number, 0.1)
                | pynutil.add_weight(graph_decimal_unit, 0.1)
            )
        else:
            self.graph = graph_number_unit | graph_percentage | pynutil.add_weight(graph_unit_number, 0.1)

        final_graph = self.add_tokens(self.graph)
        self.fst = final_graph.optimize()
