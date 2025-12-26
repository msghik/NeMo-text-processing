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

from nemo_text_processing.inverse_text_normalization.fa.verbalizers.cardinal import CardinalFst
from nemo_text_processing.inverse_text_normalization.fa.verbalizers.date import DateFst
from nemo_text_processing.inverse_text_normalization.fa.verbalizers.decimal import DecimalFst
from nemo_text_processing.inverse_text_normalization.fa.verbalizers.fraction import FractionFst
from nemo_text_processing.inverse_text_normalization.fa.verbalizers.measure import MeasureFst
from nemo_text_processing.inverse_text_normalization.fa.verbalizers.money import MoneyFst
from nemo_text_processing.inverse_text_normalization.fa.verbalizers.time import TimeFst
from nemo_text_processing.inverse_text_normalization.fa.verbalizers.word import WordFst
from nemo_text_processing.text_normalization.fa.graph_utils import GraphFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars for Persian ITN.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    """

    def __init__(self):
        super().__init__(name="verbalize", kind="verbalize")

        cardinal = CardinalFst()
        cardinal_graph = cardinal.fst

        decimal = DecimalFst()
        decimal_graph = decimal.fst

        fraction = FractionFst()
        fraction_graph = fraction.fst

        time = TimeFst()
        time_graph = time.fst

        date = DateFst()
        date_graph = date.fst

        money = MoneyFst()
        money_graph = money.fst

        measure = MeasureFst()
        measure_graph = measure.fst

        word = WordFst()
        word_graph = word.fst

        graph = (
            cardinal_graph
            | decimal_graph
            | fraction_graph
            | time_graph
            | date_graph
            | money_graph
            | measure_graph
            | word_graph
        )

        self.fst = graph.optimize()
