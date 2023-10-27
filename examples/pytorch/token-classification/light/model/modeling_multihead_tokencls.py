import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import (
    BERT_START_DOCSTRING, 
    BERT_INPUTS_DOCSTRING, 
    BertModel,
)
from transformers.file_utils import (
    add_start_docstrings,
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward
)
from .modeling_output_multihead_tokencls import TokenClassifierMultiheadOutput

_CONFIG_FOR_DOC = "MultiheadClsConfig"
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "dbmdz/bert-large-cased-finetuned-conll03-english"
_TOKEN_CLASS_EXPECTED_OUTPUT = (
    "['O', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC'] "
)
_TOKEN_CLASS_EXPECTED_LOSS = 0.01


@add_start_docstrings(
    """
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BERT_START_DOCSTRING,
)
class MultiheadBertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        if hasattr(config, "name_num_labels"):
            self.name_num_labels = config.name_num_labels
            self.name_head = nn.Linear(config.hidden_size, config.name_num_labels)
        if hasattr(config, "date_num_labels"):
            self.date_num_labels = config.date_num_labels
            self.date_head = nn.Linear(config.hidden_size, config.date_num_labels)

        self.training_part = config.training_part if hasattr(config, "training_part") else None

        # Initialize weights and apply final processing
        self.post_init()

    @property
    def with_name_head(self):
        return self.training_part == "name"
    
    @property
    def with_date_head(self):
        return self.training_part == "date"
    
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_TOKEN_CLASSIFICATION,
        output_type=TokenClassifierMultiheadOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_TOKEN_CLASS_EXPECTED_OUTPUT,
        expected_loss=_TOKEN_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierMultiheadOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        # choice name or date head
        if self.with_name_head:
            name_logits = self.name_head(sequence_output)
        if self.with_date_head:
            date_logits = self.date_head(sequence_output)
        logits = name_logits if self.with_name_head else date_logits

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierMultiheadOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    