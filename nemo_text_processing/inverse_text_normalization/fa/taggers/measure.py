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


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measurements in inverse text normalization.
    Converts spoken form to written form, e.g.
        "پنجاه درصد" -> measure { cardinal { integer: "50" } units: "%" }
        "صد کیلوگرم" -> measure { cardinal { integer: "100" } units: "kg" }
        "بیست و پنج درجه سانتی‌گراد" -> measure { cardinal { integer: "25" } units: "°C" }

    Args:
        itn_cardinal_tagger: cardinal ITN tagger
        itn_decimal_tagger: decimal ITN tagger (optional)
        itn_fraction_tagger: fraction ITN tagger (optional)
        deterministic: if True will provide a single transduction option
    """

    def __init__(
        self,
        itn_cardinal_tagger,
        itn_decimal_tagger=None,
        itn_fraction_tagger=None,
        deterministic: bool = True,
    ):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)

        cardinal_graph = itn_cardinal_tagger.graph

        # Unit mapping (spoken -> written symbol)
        unit_map = pynini.string_map(
            [
                ("درصد", "%"),
                ("کیلوگرم", "kg"),
                ("گرم", "g"),
                ("میلی‌گرم", "mg"),
                ("کیلومتر", "km"),
                ("متر", "m"),
                ("سانتی‌متر", "cm"),
                ("میلی‌متر", "mm"),
                ("کیلومتر مربع", "km²"),
                ("متر مربع", "m²"),
                ("سانتی‌متر مربع", "cm²"),
                ("لیتر", "l"),
                ("میلی‌لیتر", "ml"),
                ("کیلومتر بر ساعت", "km/h"),
                ("متر بر ثانیه", "m/s"),
                ("درجه سانتی‌گراد", "°C"),
                ("درجه فارنهایت", "°F"),
            ]
        )

        # Cardinal component
        graph_cardinal = pynutil.insert('cardinal { integer: "') + cardinal_graph + pynutil.insert('" }')

        # Units
        graph_units = pynutil.insert('units: "') + unit_map + pynutil.insert('"')

        # Measure: cardinal + units
        final_graph = graph_cardinal + pynutil.delete(" ") + insert_space + graph_units

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
