# transformer
This code implements a machine translation model from scratch using Transformer.
The source language is German and target language is English.

# train & test
You can run this code with simple command.
```python3 main.py```

# argument
You can specify arguments for Transformer. For example, if you want to specify learning rate value:

```python3 main.py --learning_rate 1e-4```

### types of argument.
```
  --emb_dim: size of embding dimension of Transformer
  --ffn_dim: size of intermediate layer of position-wise feed forward network
  --num_attention_heads: number of attention heads
  --attention_drop_out: drop out rate of attention weight
  --drop_out: drop out rate of attention output
  --max_position: maximum length of input
  --num_encoder_layers: number of encoder layer
  --num_decoder_layers: number of decoder layer
  --batch_size: batch size
  --learning_rate: learning rate
  --n_epochs: number of epochs
  --gradient_clip: gradient clip value
```

# requirments
You can set requirments with ```pip install -r requirements.txt```
