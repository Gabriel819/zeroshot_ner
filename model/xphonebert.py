import torch
import torch.nn as nn
import warnings
from transformers import AutoModel
from collections import defaultdict, Counter
from text2phonemesequence import Text2PhonemeSequence
from transformers import AutoTokenizer
from xphonebert_config import config
warnings.filterwarnings("ignore")


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


class XPhoneBertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size) # (514, 768)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class XPhoneBert(nn.Module):
    def __init__(self, config, number_of_classes):
        super(XPhoneBert, self).__init__()

        self.xphonebert = AutoModel.from_pretrained("vinai/xphonebert-base")
        self.fc = nn.Linear(768, number_of_classes)

    def forward(self, x, attn_mask=None):
        x = self.xphonebert(x, attn_mask).last_hidden_state
        x = x[:, 0, :]
        x = self.fc(x)
        return x


class XPhoneBertForNER(nn.Module):
    def __init__(self, config, number_of_classes):
        super(XPhoneBertForNER, self).__init__()

        self.xphonebert = AutoModel.from_pretrained("vinai/xphonebert-base")
        # replace embeddings to custom embeddings
        embedding_state_dict = self.xphonebert.embeddings.state_dict()
        self.xphonebert.embeddings = XPhoneBertEmbeddings(self.xphonebert.config)
        self.xphonebert.embeddings.load_state_dict(embedding_state_dict, strict=False)
        print("replaced embeddings to xphonebert embeddings")
        if config["freeze_encoder"]:
            print("Freezeing the pretrained backbone!")
            for name, param in self.xphonebert.named_parameters():
                if "embeddings" not in name:
                    param.requires_grad = False

        # TODO: universal ipa projection...
        if 'context_attn_head' in config:
            if config['context_attn_head'] is not None:
                self.word_attn = nn.MultiheadAttention(768, num_heads=config['context_attn_head'], dropout=0.1,
                                                       batch_first=True)
            else:
                self.word_attn = None
        else:
            self.word_attn = None

        self.fc = nn.Linear(768, number_of_classes)
        self.config = config

    def forward(self, x, attn_mask=None): # x: (b_s, max_seq), attn_mask: (1, 50)
        # import pdb; pdb.set_trace()
        original_shape = x.size() # (1, 35)
        b = original_shape[0]
        max_len = original_shape[-1]
        input_ids = x.detach()

        # make sure input to the model is 2D tensor
        x = x.view(-1, max_len) # (b_s, max_seq)
        if attn_mask is not None:
            attn_mask = attn_mask.view(-1, max_len)

        # pass model
        x = self.xphonebert(x, attn_mask).last_hidden_state # (1, 50, 768)

        # store hidden dimension
        d = x.size(-1) # d: 768

        # this is when each word token (not character token) is passed through the model
        # to use their cls token to predict
        if len(original_shape) == 3: # original_shape: (1, 50)
            x = x[:, 0, :]
            attn_mask = attn_mask.view(b, -1, max_len)
        x = x.view(b, -1, d) # x: (b_s, 50, 768)
        if self.word_attn is not None: # None
            if self.config['repr'] == 'subtok':
                # 1. mask everything except for <s> and </s>
                # also attend to eos_token (sentence boundary)
                # mask_idx = ((input_ids != 0)&(input_ids != 2)).nonzero() # mask all tokens that are not cls tokens : (b,3) 이어야 함...
                # 2. mask only the paddings
                mask_idx = (input_ids == 1).nonzero()

                word_attn_mask = torch.zeros(self.word_attn.num_heads * b, attn_mask.size(1), attn_mask.size(1),
                                             device='cuda')
                word_attn_mask = word_attn_mask.to(bool)
                for head in range(1, self.word_attn.num_heads + 1):
                    word_attn_mask[head * mask_idx[:, 0], mask_idx[:, 1], mask_idx[:, 1]] = True
            else:
                # find dummy tokens that are completely masked (attn mask shape: b, max_token_len(in this batch), max_len)
                mask_idx = (attn_mask == torch.zeros(max_len, device='cuda')).nonzero()

                masked_rows = list(zip(mask_idx[:, 0].tolist(), mask_idx[:, 1].tolist()))
                padded_token_starts = defaultdict(list)
                for i, j in Counter(masked_rows).items():
                    # if all input_ids are masked, that token is a dummy token (full of pad tokens)
                    if j == max_len:
                        padded_token_starts[i[0]].append(i[1])

                word_attn_mask = torch.zeros(self.word_attn.num_heads * b, attn_mask.size(1), attn_mask.size(1),
                                             device='cuda')
                word_attn_mask = word_attn_mask.to(bool)
                for i in padded_token_starts:
                    pad_start = min(padded_token_starts[i])
                    # mask out from where that pad token starts
                    for head in range(1, self.word_attn.num_heads + 1):
                        word_attn_mask[i * head, pad_start:, pad_start:] = True

            x, _ = self.word_attn(x, x, x, attn_mask=word_attn_mask)

        feature = x
        x = self.fc(x)

        return_dict = {"logits": x, 'feature': feature}
        if self.config['repr'] == 'subtok':
            return_dict['char_feature'] = None
            return_dict['cls_tokens'] = cls_tokens.detach()

        return return_dict
