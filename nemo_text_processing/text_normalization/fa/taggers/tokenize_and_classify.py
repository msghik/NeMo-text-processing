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

import os

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.fa.graph_utils import (
    NEMO_CHAR,
    NEMO_DIGIT,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)
from nemo_text_processing.text_normalization.fa.taggers.cardinal import CardinalFst
from nemo_text_processing.text_normalization.fa.taggers.decimal import DecimalFst
from nemo_text_processing.text_normalization.fa.taggers.ordinal import OrdinalFst
from nemo_text_processing.text_normalization.fa.taggers.fraction import FractionFst
from nemo_text_processing.text_normalization.fa.taggers.time import TimeFst
from nemo_text_processing.text_normalization.fa.taggers.date import DateFst
from nemo_text_processing.text_normalization.fa.taggers.money import MoneyFst
from nemo_text_processing.text_normalization.fa.taggers.measure import MeasureFst
from nemo_text_processing.text_normalization.fa.taggers.word import WordFst
from nemo_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst
from nemo_text_processing.utils.logging import logger


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars for Persian (Farsi).
    This class can process an entire sentence.
    For deployment, this grammar will be compiled and exported to OpenFst Finite State Archive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Supported semiotic classes:
    - Cardinal numbers: "123" -> "صد و بیست و سه"
    - Decimal numbers: "3.14" -> "سه ممیز چهارده صدم"
    - Ordinal numbers: "1st", "1م" -> "اول"
    - Fractions: "1/2" -> "یک دوم"
    - Time: "3:30" -> "ساعت سه و سی دقیقه"
    - Date: "1402/05/15" -> "پانزدهم مرداد هزار و چهارصد و دو"
    - Money: "$100", "1000 تومان" -> "صد دلار", "هزار تومان"
    - Measure: "50%", "100kg" -> "پنجاه درصد", "صد کیلوگرم"

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
    """

    def __init__(
        self,
        input_case: str,
        deterministic: bool = True,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify", kind="classify", deterministic=deterministic)

        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            whitelist_file = os.path.basename(whitelist) if whitelist else ""
            far_file = os.path.join(
                cache_dir, f"_{input_case}_fa_tn_{deterministic}_deterministic{whitelist_file}.far"
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            no_digits = pynini.closure(pynini.difference(NEMO_CHAR, NEMO_DIGIT))
            self.fst_no_digits = pynini.compose(self.fst, no_digits).optimize()
            logger.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logger.info(f"Creating ClassifyFst grammars for Persian. This might take some time...")

            # Initialize taggers
            self.cardinal = CardinalFst(deterministic=deterministic)
            cardinal_graph = self.cardinal.fst

            self.decimal = DecimalFst(cardinal=self.cardinal, deterministic=deterministic)
            decimal_graph = self.decimal.fst

            self.ordinal = OrdinalFst(cardinal=self.cardinal, deterministic=deterministic)
            ordinal_graph = self.ordinal.fst

            self.fraction = FractionFst(cardinal=self.cardinal, deterministic=deterministic)
            fraction_graph = self.fraction.fst

            self.time = TimeFst(cardinal=self.cardinal, deterministic=deterministic)
            time_graph = self.time.fst

            self.date = DateFst(cardinal=self.cardinal, deterministic=deterministic)
            date_graph = self.date.fst

            self.money = MoneyFst(cardinal=self.cardinal, deterministic=deterministic)
            money_graph = self.money.fst

            self.measure = MeasureFst(
                cardinal=self.cardinal, decimal=self.decimal, deterministic=deterministic
            )
            measure_graph = self.measure.fst

            word_graph = WordFst(deterministic=deterministic).fst
            punct_graph = PunctuationFst(deterministic=deterministic).fst

            # Classify with weights (lower weight = higher priority)
            classify = (
                pynutil.add_weight(time_graph, 1.0)
                | pynutil.add_weight(date_graph, 1.0)
                | pynutil.add_weight(money_graph, 1.0)
                | pynutil.add_weight(measure_graph, 1.0)
                | pynutil.add_weight(ordinal_graph, 1.05)
                | pynutil.add_weight(fraction_graph, 1.05)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(cardinal_graph, 1.1)
            )

            # Word has lowest priority (highest weight)
            classify |= pynutil.add_weight(word_graph, 100)

            # Handle punctuation
            punct = pynutil.insert("tokens { ") + pynutil.add_weight(punct_graph, weight=1.1) + pynutil.insert(" }")
            token = pynutil.insert("tokens { ") + classify + pynutil.insert(" }")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" ")) + token + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure(pynutil.add_weight(delete_extra_space, 1.1) + token_plus_punct)
            graph = delete_space + graph + delete_space

            self.fst = graph.optimize()
            no_digits = pynini.closure(pynini.difference(NEMO_CHAR, NEMO_DIGIT))
            self.fst_no_digits = pynini.compose(self.fst, no_digits).optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
