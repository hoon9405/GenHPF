import sys
import re
import math
from typing import Union, List, Dict, Callable

import numpy as np

from transformers import PreTrainedTokenizerBase
from datasets.formatting.formatting import LazyBatch, LazyRow

__all__ = [
    "Compose",
    "Textualize",
    "Linearize",
    "Flatten",
    "TextBasedTokenize",
    "ToHierarchicalGenHPFStyle",
    "ToFlattenedGenHPFStyle"
]

class Compose:
    """Composes serveral transforms together.

    Args:
        transforms (list of Callable transform objects): list of transforms to compose.
    """
    def __init__(self, transforms: List):
        self.transforms = transforms
    
    def __call__(self, sample: Dict):
        for t in self.transforms:
            sample.update(t(sample))
        return sample
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

from typing import Dict

class Textualize:
    """Converts an event dictionary to textual representation.
    
    This converts each event dictionary existed in a HuggingFace-formatted ESDS-compliant dataset to
    its corresponding textual representation.

    Args:
        exclude_codes (list of codes): list of codes to be excluded in processing.
        return_type_ids (bool): whether to return unique tables and columns to calculate token type
            ids later in TextBasedTokenize(). Note that setting this option on without
            TextBasedTokenized() may raise an error when using `batched=True`.
    """

    def __init__(
        self,
        exclude_codes: List = None,
        return_type_ids: bool = False
    ) -> None:
        self.exclude_codes = exclude_codes if exclude_codes is not None else []
        self.return_type_ids = return_type_ids

    def __call__(self, sample: Union[LazyRow, LazyBatch]):
        """
        Args:
            sample (LazyRow or LazyBatch): A sample following ESGPT data schema. LazyRow if
                batched==False, otherwise LazyBatch.
        
        Returns:
            Dict: a row (when batched==False) or a batch (when batched==True) of ESGPT data with
                converted events
        """
        if isinstance(sample, LazyBatch):
            batched = True
        else:
            batched = False

        total_unique_tables = set()
        total_unique_columns = set()

        def _event_to_str(event: Dict):
            if isinstance(event, str):
                return event

            event_text = ""
            for key, value in event.items():
                if key == "code":
                    event_item = value.split("/")
                    if value in self.exclude_codes or event_item[0] in self.exclude_codes:
                        return None
                    if event_item[1] == "NONE":
                        continue
                    if event_item[0] not in total_unique_tables:
                        total_unique_tables.add(event_item[0])
                    event_text += " ".join(event_item) + " "
                # modifiers? metadata?
                elif key == "modifiers":
                    for modifiers_key, modifiers_value in value.items():
                        if modifiers_value is not None:
                            if modifiers_key not in total_unique_columns:
                                total_unique_columns.add(modifiers_key)
                            if type(modifiers_value) == float:
                                # round up to four decimal places
                                modifiers_value = f"{modifiers_value:.4f}".rstrip("0").rstrip(".")
                            event_text += f"{modifiers_key} {modifiers_value} "
                elif value is not None:
                    if type(value) == float:
                        # round up to four decimal places
                        value = f"{value:.4f}".rstrip("0").rstrip(".")
                    if key not in total_unique_columns:
                        total_unique_columns.add(key)
                    event_text += f"{key} {value} "
            
            if event_text == "":
                return None
            # remove trailing white space
            return event_text[:-1]

        if not batched:
            sample["static_measurements"] = [sample["static_measurements"]]
            sample["events"] = [sample["events"]]

        static_measurements_text = []
        for static_measurements_row in sample["static_measurements"]:
            static_measurements_row_text = []
            if static_measurements_row is not None:
                for static_event in static_measurements_row:
                    static_event_text = _event_to_str(static_event)
                    if static_event_text is not None:
                        static_measurements_row_text.append(static_event_text)
                if len(static_measurements_row_text) > 0:
                    static_measurements_text.append(static_measurements_row_text)
                else:
                    static_measurements_text.append(None)
            else:
                static_measurements_text.append(None)

        intimes = []
        events_text = []
        for events_row in sample["events"]:
            events_row_text = []
            intime_found = False
            if events_row is None:
                events_text.append([])
                intimes.append(None)
                continue

            for event_sequence in events_row:
                event_sequence_text = []
                # to cover flattened schema
                if not isinstance(event_sequence["measurements"], list):
                    event_sequence["measurements"] = [event_sequence["measurements"]]
                for j, event in enumerate(event_sequence["measurements"]):
                    event_text = _event_to_str(event)
                    if event["code"] == "event_type/ICU_STAY_START":
                        intimes.append(event_sequence["time"])
                        intime_found = True

                    if event_text is not None:
                        event_sequence_text.append(event_text)

                if len(event_sequence_text) > 0:
                    event_sequence["measurements"] = event_sequence_text
                    events_row_text.append(event_sequence)
            events_text.append(events_row_text)
            if not intime_found:
                if len(events_row_text) == 0:
                    intimes.append(None)
                else:
                    intimes.append(events_row_text[0]["time"])

        ret_dict = {
            "static_measurements": static_measurements_text if batched else static_measurements_text[0],
            "icustay_start_time": intimes,
            "events": events_text if batched else events_text[0],
        }
        if self.return_type_ids:
            ret_dict["total_unique_tables"] = list(total_unique_tables)
            ret_dict["total_unique_columns"] = list(total_unique_columns)

        return ret_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class Linearize:
    """linearize sequences of events to be a list of sequental events.

    This converts a nested structure of the events where each event is centered by its timestamp
    (events having the same timestamp are centralized to an event sequence) to a linearized
    structure where events are not centered anymore.
    """
    def __init__(self) -> None:
        pass

    def __call__(self, sample: Union[LazyRow, LazyBatch]):
        """
        Args:
            sample (LazyRow or LazyBatch): A sample following ESGPT data schema. LazyRow if
                batched==False, otherwise LazyBatch.
        
        Returns:
            Dict: a row (when batched==False) or a batch (when batched==True) of ESGPT data with
                flattened events
        """
        if isinstance(sample, LazyBatch):
            batched = True
        else:
            batched = False

        if not batched:
            sample["events"] = [sample["events"]]

        events_flattened = []
        for events_row in sample["events"]:
            events_row_flattened = []
            events_row = sorted(events_row, key=lambda x: x["time"])
            for event_sequence in events_row:
                if not isinstance(event_sequence["measurements"], List):
                    events_flattened.append(event_sequence)
                else:
                    for event in event_sequence["measurements"]:
                        events_row_flattened.append({
                            "measurements": event,
                            "time": event_sequence["time"]
                        })
            events_flattened.append(events_row_flattened)

        return {"events": events_flattened if batched else events_flattened[0]}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

# include Textualize() and Linearize()
class Flatten:
    """Flatten a list of textual events representations to be a single text by joining them.
    
    This joins a list of events strings so that they can be represented as a single long sequence
    of events string.
    
    Args:
        time_interval_bins (list of integers): list of integers that specify time interval bins to
            discretize time interval between subsequent events. Note that each bin is defined as
            `time_interval_bins[i]` <= x < `time_interval_bins[i+1]`. This argument is used when
            concatenating a bucketized time interval for each event as a special token to indicate
            the time gap between each subsequent events. If set to `None`, the events texts are
            concatenated as it is without these special tokens of time intervals. 
            `NOTE`: deprecated.
    
    Note:
        This transform assumes that `sample["events"]` is a list of strings (i.e., after applying 
        `transforms.Textualize(...)` and `transforms.Linearize(...)`).
    """

    def __init__(
        self,
        time_interval_bins: List = [0, 1, 5, 10, 15, 20, 30, 40, 60, 61, 120, sys.maxsize]
    ):
        self.time_interval_bins = time_interval_bins

    def __call__(self, sample: Union[LazyRow, LazyBatch]):
        """
        Args:
            sample (LazyRow or LazyBatch): A sample following ESGPT data schema. LazyRow if
                batched==False, otherwise LazyBatch.
        
        Returns:
            Dict: a row (when batched==False) or a batch (when batched==True) of ESGPT data with
                a long sequence of string for representing events.
        """
        if isinstance(sample, LazyBatch):
            batched = True
        else:
            batched = False

        if not batched:
            sample["events"] = [sample["events"]]
            sample["static_measurements"] = [sample["static_measurements"]]

        if len(sample["events"][0]) > 0:
            assert isinstance(sample["events"][0][0]["measurements"], str), (
                "Flatten() must be applied after transforms.Textualize() "
                "and transforms.Linearize(). Please ensure that these two transforms "
                "functions are composed before transforms.Flatten()."
            )

        static_measurements_linearized = []
        if sample["static_measurements"] is None or all([x is None for x in sample["static_measurements"]]):
            static_measurements_linearized = None
        else:
            for static_measurements_row in sample["static_measurements"]:
                static_measurements_row_linearized = " ".join(static_measurements_row)
                static_measurements_linearized.append(static_measurements_row_linearized)

        events_linearized = []
        for events_row in sample["events"]:
            events_row_linearized = ""
            for event in events_row:
                events_row_linearized += event["measurements"] + " [SEP] "
            events_linearized.append(events_row_linearized.rstrip("\[SEP\] "))

        if static_measurements_linearized is not None and not batched:
            static_measurements_linearized = static_measurements_linearized[0]

        ret_dict = {
            "static_measurements": static_measurements_linearized,
            "events": events_linearized if batched else events_linearized[0],
        }

        if static_measurements_linearized is None:
            del ret_dict["static_measurements"]
        
        return ret_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

# tokenize
class TextBasedTokenize:
    """Tokenizes textual events representations so that they can be proceeded to DL models.
    
    Args:
        tokenizer (str or HuggingFace Tokenizer): a string indicating a specific tokenizer from
            HuggingFace tokenizers, or an instance for one of them. Note that we do not allow
            any other tokenizers outside HuggingFace tokenizers.
        max_num_events (int): an integer value to indicate max number of events. If a sample
            contains events more than this value, it is split into a list of event sequences by
            repeatedly truncating every `max_num_events`-length events without overlapping.
            Note that this functionality is only activated When a sample has a hierarchical structure
            where each subject has a list of events.
        max_sequence_length (int): an integer value to indicate max length of tokens. If a resulted
            token sequence is longer than this value, it is split into a list of token sequences
            by repeatedly truncating every `max_sequence_length`-length tokens without overlapping.
            Note that this functionality is only activated when a sample has a flattened structure
            where an event sequence of a single subject is represented as a long single sentence,
            which is likely to have a very long sequence of tokens. 
    
    Note:
        We highly recommened to map this transform function with `batched=True` for taking two
        advantages; 1) to speed up processing tokenization with batches, and 2) to control the
        size of the output dataset where each truncated sequence of tokens now can be a single
        data sample to be processed. In other words, if it is called with batch mapping, then
        each truncated tokens sequence (`max_sequence_length`-length tokens or `max_num_events`
        -length events) is treated as a single data point. Otherwise, it remains with a list of
        sequences where you may need to further process to handle them (e.g., random select one
        of them every iteration in the data loader).
    """
    def __init__(
        self,
        tokenizer: Union[str, Callable] = None,
        max_num_events: int = 256,
        max_sequence_length: int = 8192,
        return_type_ids: bool = False,
        return_dpe_ids: bool = False,
    ):
        if tokenizer is not None and isinstance(tokenizer, Callable):
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer

            if isinstance(tokenizer, str):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            else:
                # default (when tokenizer is None)
                self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            
            self.special_tokens = {}
        
        assert isinstance(self.tokenizer, PreTrainedTokenizerBase), (
            "The argument, `tokenizer`, must be an instance of huggingface tokenizer "
            "(PreTrainedTokenizerBase) or a string specification of any huggingface tokenizers."
        )

        self.max_num_events = max_num_events
        self.max_sequence_length = max_sequence_length
        self.return_type_ids = return_type_ids
        self.return_dpe_ids = return_dpe_ids
        
    def __call__(self, sample: Union[LazyRow, LazyBatch]):
        """
        Args:
            sample (LazyRow or LazyBatch): A sample following ESGPT data schema. LazyRow if
                batched==False, otherwise LazyBatch.
        """

        if isinstance(sample, LazyBatch):
            batched = True
        else:
            batched = False

        self.total_unique_tables = None
        if "total_unique_tables" in sample:
            self.total_unique_tables = sample["total_unique_tables"]
            del sample["total_unique_tables"]
        self.total_unique_columns = None
        if "total_unique_columns" in sample:
            self.total_unique_columns = sample["total_unique_columns"]
            del sample["total_unique_columns"]

        if sample["static_measurements"] is None or all([x is None for x in sample["static_measurements"]]):
            static_measurements = None
        else:
            static_measurements = self.tokenize(sample["static_measurements"])

        if not batched:
            sample["events"] = [sample["events"]]

        sample["events"] = [s if len(s) > 0 else None for s in sample["events"]]

        depth = self.infer_depth(sample["events"])
        match depth:
            case 2: # for flattened structure
                events_tokenized = self.tokenize(sample["events"])
                for i in range(len(events_tokenized)):
                    if events_tokenized[i] is not None and len(events_tokenized[i]["input_ids"]) > self.max_sequence_length:
                        start_idx = -self.max_sequence_length
                        for offset, token_type in enumerate(events_tokenized[i]["token_type_ids"][start_idx:]):
                            if token_type == 6: # [SEP]
                                break
                        start_idx += offset
                        events_tokenized[i]["input_ids"][start_idx] = 101 # [SEP] (102) -> [CLS] (101)
                        events_tokenized[i]["input_ids"] = events_tokenized[i]["input_ids"][start_idx:]
                        events_tokenized[i]["token_type_ids"][start_idx] = 5 # [SEP] (6) -> [CLS] (5)
                        events_tokenized[i]["token_type_ids"] = events_tokenized[i]["token_type_ids"][start_idx:]
                        events_tokenized[i]["dpe_ids"] = events_tokenized[i]["dpe_ids"][start_idx:]

            case 4: # for hierarchical structure
                events_tokenized = self.tokenize(sample["events"])
                events_tokenized = [x[-self.max_num_events:] if x is not None else None for x in events_tokenized]

            # case 4: # for nested structure
            #     events_tokenized = self.tokenize(sample["events"])

        ret_dict = {
            "events": events_tokenized,
            "static_measurements": static_measurements
        }

        if ret_dict["static_measurements"] is None:
            del ret_dict["static_measurements"]

        return ret_dict

    def infer_depth(self, list_or_dict):
        if isinstance(list_or_dict, list):
            return 1 + max(self.infer_depth(item) for item in list_or_dict)
        elif isinstance(list_or_dict, dict):
            return 1 + max(self.infer_depth(item) for item in list_or_dict.values())
        else:
            return 1

    def _tokenize(self, text: str):
        number_groups = [g for g in re.finditer(r"([0-9]+([.][0-9]*)?|[0-9]+|\.+)", text)]
        # split digit place (e.g., "123.456" -> "1 2 3 . 4 5 6")
        text = re.sub(r"([0-9\.])", r" \1 ", text)
        text = "[CLS] " + text + " [SEP]"
        tokenized = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            max_length=None,
            truncation=False,
            return_overflowing_tokens=False,
        )
        if self.return_dpe_ids:
            numbers = [i for i, j in enumerate(tokenized.input_ids) if j in list(range(121, 131)) + [119]]
            numbers_cnt = 0
            dpe_ids = [0] * len(tokenized.input_ids)
            for group in number_groups:
                if group[0] == "." * len(group[0]):
                    numbers_cnt += len(group[0])
                    continue
                
                start = numbers[numbers_cnt]
                end = numbers[numbers_cnt + len(group[0]) - 1] + 1
                matched_numbers = tokenized.input_ids[start:end]
                digits = [i for i, j in enumerate(matched_numbers) if j == 119]

                # in case of integer
                if len(digits) == 0:
                    dpe_ids[start:end] = list(range(len(group[0]) + 5, 5, -1))
                # in case of float
                elif len(digits) == 1:
                    digit_idx = len(group[0]) - digits[0]
                    dpe_ids[start:end] = list(range(len(group[0]) + 5 - digit_idx, 5 - digit_idx, -1))
                else:
                    raise ValueError()
                
                numbers_cnt += len(group[0])
        else:
            dpe_ids = None
        
        # get token type embedding
        if self.return_type_ids:
            table_offsets = [
                g.span() for g in re.finditer("|".join(self.total_unique_tables), text)
            ]
            column_offsets = [
                g.span() for g in re.finditer("|".join(self.total_unique_columns), text)
            ]
            sep_offsets = [
                g.span() for g in re.finditer("\[SEP\]", text)
            ]
            type_ids = [3] * len(tokenized.input_ids)
            type_ids[0] = 5 # [CLS]
            type_ids[-1] = 6
            for table_offset in table_offsets:
                for i, offset in enumerate(tokenized.offset_mapping):
                    if offset[0] == offset[1]:
                        continue
                    if table_offset[0] <= offset[0] and offset[1] <= table_offset[1]:
                        type_ids[i] = 1

                    if table_offset[1] < offset[1]:
                        break
            
            for column_offset in column_offsets:
                for i, offset in enumerate(tokenized.offset_mapping):
                    if offset[0] == offset[1]:
                        continue
                    if column_offset[0] <= offset[0] and offset[1] <= column_offset[1]:
                        type_ids[i] = 2
                    
                    if column_offset[1] < offset[1]:
                        break
            for sep_offset in sep_offsets:
                for i, offset in enumerate(tokenized.offset_mapping):
                    if offset[0] == offset[1]:
                        continue
                    if sep_offset[0] <= offset[0] and offset[1] <= sep_offset[1]:
                        type_ids[i] = 6
                    
                    if sep_offset[1] < offset[1]:
                        break
        else:
            type_ids = None

        return {
            "input_ids": tokenized.input_ids,
            "token_type_ids": type_ids,
            "dpe_ids": dpe_ids
        }

    def tokenize(self, list_or_dict):
        if isinstance(list_or_dict, list):
            depth = self.infer_depth(list_or_dict)
            if depth <= 1:
                return self._tokenize(list_or_dict)
            else:
                return [self.tokenize(item) for item in list_or_dict]
        elif isinstance(list_or_dict, dict):
            return {k: self.tokenize(v) for k, v in list_or_dict.items()}
        else:
            if isinstance(list_or_dict, str):
                return self._tokenize(list_or_dict)
            else:
                return list_or_dict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class ToHierarchicalGenHPFStyle:
    """Converts the ESDS-compliant dataset to follow the hierarchical GenHPF style.
    
    This transform is equivalent with a sequence of corresponding transforms functions as follows:
    transforms.Textualize() --> transforms.Linearize() --> transforms.TextBasedTokenize()

    Args:
        exclude_codes (list of codes): list of codes to be excluded in processing.
        tokenizer (str or HuggingFace Tokenizer): a string indicating a specific tokenizer from
            HuggingFace tokenizers, or an instance for one of them. Note that we do not allow
            any other tokenizers outside HuggingFace tokenizers.
        max_num_events (int): an integer value to indicate max number of events. If a sample
            contains events more than this value, it is split into a list of event sequences by
            repeatedly truncating every `max_num_events`-length events without overlapping.
            Note that this functionality is only activated When a sample has a hierarchical structure
            where each subject has a list of events.
        max_sequence_length (int): an integer value to indicate max length of tokens. If a resulted
            token sequence is longer than this value, it is split into a list of token sequences
            by repeatedly truncating every `max_sequence_length`-length tokens without overlapping.
            Note that this functionality is only activated when a sample has a flattened structure
            where an event sequence of a single subject is represented as a long single sentence,
            which is likely to have a very long sequence of tokens. 
    """
    def __init__(
        self,
        exclude_codes: List = None,
        tokenizer: Union[str, Callable] = None,
        max_num_events: int = 256,
        max_sequence_length: int = 1024,
        return_type_ids: bool = False,
        return_dpe_ids: bool = False,
    ):
        self.preliminary_transform = Compose([
            Textualize(exclude_codes, return_type_ids=return_type_ids),
            Linearize(),
            TextBasedTokenize(
                tokenizer=tokenizer,
                max_num_events=max_num_events,
                max_sequence_length=max_sequence_length,
                return_type_ids=return_type_ids,
                return_dpe_ids=return_dpe_ids
            )
        ])
    
    def __call__(self, sample: Union[LazyRow, LazyBatch]):
        return self.preliminary_transform(sample)

class ToFlattenedGenHPFStyle:
    """Converts the ESDS-compliant dataset to follow the flattened GenHPF style.
    
    This transform is equivalent with a sequence of corresponding transforms functions as follows:
    transforms.Textualize() --> transforms.Linearize() --> transforms.Flatten()
    --> transforms.TextBasedTokenize()

    Args:
        exclude_codes (list of codes): list of codes to be excluded in processing.
        time_interval_bins (list of integers): list of integers that specify time interval bins to
            discretize time interval between subsequent events. Note that each bin is defined as
            `time_interval_bins[i]` <= x < `time_interval_bins[i+1]`. This argument is used when
            concatenating a bucketized time interval for each event as a special token to indicate
            the time gap between each subsequent events. If set to `None`, the events texts are
            concatenated as it is without these special tokens of time intervals.
            `NOTE`: deprecated.
        tokenizer (str or HuggingFace Tokenizer): a string indicating a specific tokenizer from
            HuggingFace tokenizers, or an instance for one of them. Note that we do not allow
            any other tokenizers outside HuggingFace tokenizers.
        max_num_events (int): an integer value to indicate max number of events. If a sample
            contains events more than this value, it is split into a list of event sequences by
            repeatedly truncating every `max_num_events`-length events without overlapping.
            Note that this functionality is only activated When a sample has a hierarchical structure
            where each subject has a list of events.
        max_sequence_length (int): an integer value to indicate max length of tokens. If a resulted
            token sequence is longer than this value, it is split into a list of token sequences
            by repeatedly truncating every `max_sequence_length`-length tokens without overlapping.
            Note that this functionality is only activated when a sample has a flattened structure
            where an event sequence of a single subject is represented as a long single sentence,
            which is likely to have a very long sequence of tokens. 
    """
    def __init__(
        self,
        exclude_codes: List = None,
        time_interval_bins: List = None,
        tokenizer: Union[str, Callable] = None,
        max_num_events: int = 256,
        max_sequence_length: int = 8192,
        return_type_ids: bool = False,
        return_dpe_ids: bool = False,
    ):

        self.preliminary_transform = Compose([
            Textualize(exclude_codes=exclude_codes, return_type_ids=return_type_ids),
            Linearize(),
            Flatten(time_interval_bins=time_interval_bins),
            TextBasedTokenize(
                tokenizer=tokenizer,
                max_num_events=max_num_events,
                max_sequence_length=max_sequence_length,
                return_type_ids=return_type_ids,
                return_dpe_ids=return_dpe_ids
            )
        ])
    
    def __call__(self, sample: Union[LazyRow, LazyBatch]):
        return self.preliminary_transform(sample)