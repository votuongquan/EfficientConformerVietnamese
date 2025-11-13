# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version  2.0 (the "License");
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

# PyTorch
import torch
import torch.nn as nn
import numpy as np

# Base Model
from models.model import Model

# Encoders
from models.encoders import (
    ConformerEncoder,
    ConformerEncoderInterCTC
)

# Losses
from models.losses import (
    LossCTC, 
    LossInterCTC
)

# CTC Decode Beam Search - NOW USING pyctcdecode
from pyctcdecode import build_ctcdecoder


class ModelCTC(Model):

    def __init__(self, encoder_params, tokenizer_params, training_params, decoding_params, name):
        super(ModelCTC, self).__init__(tokenizer_params, training_params, decoding_params, name)

        # Encoder
        if encoder_params["arch"] == "Conformer":
            self.encoder = ConformerEncoder(encoder_params)
        else:
            raise Exception("Unknown encoder architecture:", encoder_params["arch"])

        # FC Layer
        self.fc = nn.Linear(
            encoder_params["dim_model"][-1] if isinstance(encoder_params["dim_model"], list) else encoder_params["dim_model"],
            tokenizer_params["vocab_size"]
        )

        # Criterion
        self.criterion = LossCTC()

        # Compile
        self.compile(training_params)

        # Store decoding hyperparams for pyctcdecode
        self.beam_size = decoding_params.get("beam_size", 5)
        self.tmp = decoding_params.get("temperature", 1.0)
        self.ngram_path = decoding_params.get("ngram_path", None)
        self.ngram_alpha = decoding_params.get("ngram_alpha", 0.0)
        self.ngram_beta = decoding_params.get("ngram_beta", 0.0)
        self.ngram_offset = decoding_params.get("ngram_offset", 0)  # usually 0 or ord('a')-1

        # Will hold the pyctcdecode decoder
        self._pyctc_decoder = None
        self._pyctc_decoder_beam_size = None

    def forward(self, batch):
        # Unpack Batch
        x, _, x_len, _ = batch

        # Forward Encoder (B, Taud) -> (B, T, Denc)
        logits, logits_len, attentions = self.encoder(x, x_len)

        # FC Layer (B, T, Denc) -> (B, T, V)
        logits = self.fc(logits)

        return logits, logits_len, attentions

    def distribute_strategy(self, rank):
        super(ModelCTC, self).distribute_strategy(rank)

        self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        self.encoder = torch.nn.parallel.DistributedDataParallel(self.encoder, device_ids=[self.rank])
        self.fc = torch.nn.parallel.DistributedDataParallel(self.fc, device_ids=[self.rank])

    def load_encoder(self, path):
        # Load Encoder Params
        checkpoint = torch.load(path, map_location=next(self.parameters()).device)
        if checkpoint["is_distributed"] and not self.is_distributed:
            self.encoder.load_state_dict({
                key.replace(".module.", "").replace("encoder.", ""): value
                for key, value in checkpoint["model_state_dict"].items()
                if key[:len("encoder")] == "encoder"
            })
        else:
            self.encoder.load_state_dict({
                key.replace("encoder.", ""): value
                for key, value in checkpoint["model_state_dict"].items()
                if key[:len("encoder")] == "encoder"
            })

        # Print Encoder state
        if self.rank == 0:
            print("Model encoder loaded at step {} from {}".format(checkpoint["model_step"], path))

    def gready_search_decoding(self, x, x_len):
        # Forward Encoder (B, Taud) -> (B, T, Denc)
        logits, logits_len = self.encoder(x, x_len)[:2]

        # FC Layer (B, T, Denc) -> (B, T, V)
        logits = self.fc(logits)

        # Softmax -> Log > Argmax -> (B, T)
        preds = logits.log_softmax(dim=-1).argmax(dim=-1)

        # Batch Pred List
        batch_pred_list = []

        # Batch loop
        for b in range(logits.size(0)):
            blank = False
            pred_list = []

            # Decoding Loop
            for t in range(logits_len[b]):
                if preds[b, t] == 0:
                    blank = True
                    continue

                if len(pred_list) == 0:
                    pred_list.append(preds[b, t].item())
                elif pred_list[-1] != preds[b, t] or blank:
                    pred_list.append(preds[b, t].item())
                
                blank = False

            batch_pred_list.append(pred_list)

        # Decode Sequences
        return self.tokenizer.decode(batch_pred_list)

    def _ensure_pyctc_decoder(self, beam_size):
        """
        Lazily build or rebuild the pyctcdecode decoder if needed.
        """
        if (self._pyctc_decoder is None or 
            self._pyctc_decoder_beam_size != beam_size):
            
            labels = [chr(idx + self.ngram_offset) for idx in range(self.tokenizer.vocab_size())]

            self._pyctc_decoder = build_ctcdecoder(
                labels=labels,
                kenlm_model_path=self.ngram_path,
                alpha=self.ngram_alpha,
                beta=self.ngram_beta,
                unigram_path=None,
                beam_width=beam_size,
                beam_cutoff_top_n=self.tokenizer.vocab_size(),
                beam_prune_logp=-10.0,
                num_processes=4,
                blank_id=0,
                log_probs_input=True,
            )
            self._pyctc_decoder_beam_size = beam_size

    def beam_search_decoding(self, x, x_len, beam_size=None):
        if beam_size is None:
            beam_size = self.beam_size

        # Ensure decoder is built with correct beam size
        self._ensure_pyctc_decoder(beam_size)

        # Forward Encoder
        logits, logits_len = self.encoder(x, x_len)[:2]
        logits = self.fc(logits)

        # Apply temperature
        logits = logits / self.tmp

        # Log probabilities (pyctcdecode expects log probs)
        log_probs = logits.log_softmax(dim=-1)  # (B, T, V)

        # Convert to numpy for pyctcdecode
        log_probs_np = log_probs.detach().cpu().numpy()
        seq_lens_np = logits_len.cpu().numpy()

        # Run beam search
        beam_results = self._pyctc_decoder.decode_batch(
            log_probs_np, seq_lens_np, beam_width=beam_size
        )

        # Extract best hypothesis (token IDs)
        batch_pred_list = []
        for res in beam_results:
            # res.tokens: list of int token indices (excluding blanks)
            batch_pred_list.append([int(tok) for tok in res.tokens])

        # Decode using original tokenizer
        return self.tokenizer.decode(batch_pred_list)


class InterCTC(ModelCTC):

    def __init__(self, encoder_params, tokenizer_params, training_params, decoding_params, name):
        # Note: super() calls ModelCTC.__init__, which now handles pyctcdecode setup
        super(ModelCTC, self).__init__(tokenizer_params, training_params, decoding_params, name)

        # Update Encoder Params
        encoder_params["vocab_size"] = tokenizer_params["vocab_size"]

        # Encoder
        if encoder_params["arch"] == "Conformer":
            self.encoder = ConformerEncoderInterCTC(encoder_params)
        else:
            raise Exception("Unknown encoder architecture:", encoder_params["arch"])

        # FC Layer
        self.fc = nn.Linear(
            encoder_params["dim_model"][-1] if isinstance(encoder_params["dim_model"], list) else encoder_params["dim_model"],
            tokenizer_params["vocab_size"]
        )

        # Criterion
        self.criterion = LossInterCTC(training_params["interctc_lambda"])

        # Compile
        self.compile(training_params)

    def forward(self, batch):
        # Unpack Batch
        x, _, x_len, _ = batch

        # Forward Encoder
        logits, logits_len, attentions, interctc_logits = self.encoder(x, x_len)

        # FC Layer
        logits = self.fc(logits)

        return logits, logits_len, attentions, interctc_logits