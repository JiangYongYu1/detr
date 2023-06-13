from this import d
from turtle import forward
from models import transformer
import torch.nn as nn
import torch

class TestModel(nn.Module):
    def __init__(self, decode_model, num_queries, hidden_dim):
        super().__init__()
        self.decode_model = decode_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

    def forward(self, tensor, mask):
        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, 1, 1)
        tgt = torch.zeros_like(query_embed)
        pos_embed = torch.rand((1681, 1, 256), device=tensor.device, dtype=tensor.dtype)

        out = self.decode_model(tgt, tensor, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return out[0]

if __name__ == "__main__":
    d_model=256
    nhead=8
    num_encoder_layers=6
    num_decoder_layers=1
    dim_feedforward=2048
    dropout=0.1
    activation="relu"
    normalize_before=False
    return_intermediate_dec=True
    decoder_layer = transformer.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
    decoder_norm = nn.LayerNorm(d_model)
    # decoder_norm = None
    decode_model = transformer.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)
    model = TestModel(decode_model, 100, d_model).cuda()
    model.eval()

    input_data = torch.rand((1681, 1, 256), dtype=torch.float32).cuda()
    mask = torch.ones((1, 1681), dtype=torch.float32).cuda()
    onnx_path = "decode.onnx"
    torch.onnx.export(
        model,  # --dynamic only compatible with cpu
        args=(input_data, mask),
        f=onnx_path,
        verbose=False,
        opset_version=11,
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        input_names=['images', 'masks'],
        # output_names=['output_class', "output_box"],
    )

    # import onnx
    # onnx_model = onnx.load(onnx_path)
   
    # dim_proto_1 = onnx_model.graph.output[0].type.tensor_type.shape.dim[1]
    # dim_proto_1.dim_param = '1'
    # onnx.save(onnx_model, 'decode.onnx')