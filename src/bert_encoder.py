from transformers.models.bert.modeling_bert import BertModel, BertEncoder, BaseModelOutputWithPastAndCrossAttentions, BertEmbeddings, BertPooler
from dataclasses import dataclass
import torch.nn as nn

@dataclass
class DetachedOutput(BaseModelOutputWithPastAndCrossAttentions):
    pass


class DetachedBertEncoder(BertEncoder):
    def __init__(self, config, detach_layer_idx=[3, 6]):
        super().__init__(config)
        self.detach_layer_idx = detach_layer_idx
        self.detach = False
        # self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        # self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):

        all_hidden_states = ()  # if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            # always return undetached layers
            if output_hidden_states:  # if true keep all states
                all_hidden_states = all_hidden_states + (hidden_states,)
            else:
                if i in self.detach_layer_idx:  # else keep only marked layers
                    all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # detach before attention layer if needed
            if self.detach and i in self.detach_layer_idx:
                hidden_states = hidden_states.detach()
            
            # gradient_checkpointing not implemented in puhti transformers version
            # if self.gradient_checkpointing and self.training:

            #     if use_cache:
            #         logger.warning(
            #             "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            #         )
            #         use_cache = False

            #     def create_custom_forward(module):
            #         def custom_forward(*inputs):
            #             return module(*inputs, past_key_value, output_attentions)

            #         return custom_forward

            #     layer_outputs = torch.utils.checkpoint.checkpoint(
            #         create_custom_forward(layer_module),
            #         hidden_states,
            #         attention_mask,
            #         layer_head_mask,
            #         encoder_hidden_states,
            #         encoder_attention_mask,
            #     )
            # else:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # if output_hidden_states:
        # always return final hidden state
        all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        # just a struct for wrapping returns
        # do we need self_attentions of hidden_states ?
        # ^ need hidden states
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class DetachedBertModel(BertModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True, detach_layer_idx=[3, 6]):
        # super(BertModel, self).__init__(config)
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = DetachedBertEncoder(config, detach_layer_idx)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        # self.post_init()
        self.init_weights()  # old library function

    def detach_encoder(self):
        self.encoder.detach = True

    def undetach_encoder(self):
        self.encoder.detach = False