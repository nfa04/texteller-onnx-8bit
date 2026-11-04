#!/usr/bin/env python3
"""Reads a picture of a formula with the two 8-bit files, as an example of how to run them (and as a smoke test).

usage: read_image.py MODEL_DIR IMAGE        MODEL_DIR holds encoder.onnx and decoder.onnx
       read_image.py MODEL_DIR --demo       draws a small formula itself and reads it

The picture is prepared the way TexTeller wants it: gray, scaled so that the longer side is 448 px (the proportions are kept),
normalised, and put in the top left corner of a 448 x 448 square that is zero elsewhere.

For HANDWRITING drawn from pen strokes, draw the line THIN (about 1 px, a fiftieth of the height of a letter): TexTeller learned
from printed formulas and reads a pen that thin far better than a thick one (see README.md).
"""
import json
import sys

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from tokenizers import Tokenizer

MEAN, STD = 0.9545467, 0.15394445
SIZE = 448
LAYERS, START, EOS = 12, 0, 2


def picture(img: Image.Image) -> np.ndarray:
    g = np.asarray(img.convert('L'), dtype=np.float32) / 255.0
    ys, xs = np.where(g < 0.94)  # trim the white border
    if len(xs):
        g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = g.shape
    s = SIZE / max(h, w)
    g = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC), dtype=np.float32) / 255.0
    out = np.zeros((SIZE, SIZE), np.float32)
    out[:g.shape[0], :g.shape[1]] = (g - MEAN) / STD
    return out[None, None]


def read(model_dir: str, img: Image.Image, max_tokens: int = 200) -> str:
    opts = ort.SessionOptions()
    enc = ort.InferenceSession(f'{model_dir}/encoder.onnx', opts, providers=['CPUExecutionProvider'])
    dec = ort.InferenceSession(f'{model_dir}/decoder.onnx', opts, providers=['CPUExecutionProvider'])
    tok = Tokenizer.from_file(hf_hub_download('OleehyO/TexTeller', 'tokenizer.json'))

    hidden = enc.run(None, {'pixel_values': picture(img)})[0]
    past = [f'past_key_values.{i}.{w}.{kv}' for i in range(LAYERS) for w in ('decoder', 'encoder') for kv in ('key', 'value')]
    present = [n.replace('past_key_values', 'present') for n in past]
    nothing = np.zeros((1, 16, 1, 64), np.float32)  # (the first step has no cache; the graph wants something there)

    feed = {'input_ids': np.array([[START]], np.int64), 'encoder_hidden_states': hidden, 'use_cache_branch': np.array([False])}
    feed.update({n: nothing for n in past})
    ids, cache = [START], None
    while len(ids) < max_tokens:
        out = dec.run(['logits'] + present, feed)
        token = int(out[0][0, -1].argmax())
        ids.append(token)
        if token == EOS:
            break
        if cache is None:
            cache = dict(zip(past, out[1:]))  # first step: the keys and values of the picture and of the first token
        else:
            for n, v in zip(past, out[1:]):
                if '.decoder.' in n:
                    cache[n] = v  # later steps only add to those of the tokens; the ones of the picture stay
        feed = {'input_ids': np.array([[token]], np.int64), 'encoder_hidden_states': hidden, 'use_cache_branch': np.array([True])}
        feed.update(cache)

    text = tok.decode([i for i in ids[1:] if i != EOS], skip_special_tokens=True).replace('Ġ', ' ').strip()
    for fence in (('\\[', '\\]'),):  # TexTeller writes every formula between \[ and \]
        if text.startswith(fence[0]):
            text = text[len(fence[0]):]
        if text.endswith(fence[1]):
            text = text[:-len(fence[1])]
    return text.strip()


def demo() -> Image.Image:
    """x^2 + 3 = 7, drawn with the default font: a big x with a small raised 2, then the rest."""
    img = Image.new('RGB', (360, 110), 'white')
    d = ImageDraw.Draw(img)
    big, small = ImageFont.load_default(size=56), ImageFont.load_default(size=30)
    d.text((20, 30), 'x', fill='black', font=big)
    d.text((55, 22), '2', fill='black', font=small)
    d.text((90, 30), '+ 3 = 7', fill='black', font=big)
    return img


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    image = demo() if sys.argv[2] == '--demo' else Image.open(sys.argv[2])
    print(read(sys.argv[1], image))
