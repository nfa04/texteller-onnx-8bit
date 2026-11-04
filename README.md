# TexTeller 3.0, 8-bit ONNX

Two small ONNX files that read a picture of a formula (printed or handwritten) and write it as LaTeX, and the script that makes
them from the published ones.

|                | size   | what it is                                                                                  |
|----------------|--------|---------------------------------------------------------------------------------------------|
| `encoder.onnx` | 88 MB  | the vision encoder                                                                          |
| `decoder.onnx` | 229 MB | the text decoder, with a key/value cache (the first step and the later steps are one graph) |

The originals are [TexTeller 3.0](https://github.com/OleehyO/TexTeller) by OleehyO and contributors
([weights](https://huggingface.co/OleehyO/TexTeller), Apache-2.0). They are float32 and 1.25 GB; these are 8-bit and 317 MB, which
fits a tablet. **This is a conversion of their work, not made or endorsed by them.**

## What was changed

1. Both networks were quantised to 8-bit weights with onnxruntime's dynamic quantisation (`QuantType.QInt8`).
2. The decoder without a cache (first step) and with a cache (later steps) were quantised separately and merged into one graph
   that shares their weights. (onnxruntime's quantiser skips the `If` branches of the published merged decoder, which would stay at
   909 MB.)

Nothing was retrained or fine-tuned. `make_models.py` does exactly this and nothing else.

## Using them

```bash
pip install -r requirements.txt && pip install optimum==2.1.0 optimum-onnx==0.1.0 --no-deps
python read_image.py out --demo            # draws "x^2 + 3 = 7" and prints  x^{2}+3=7
python read_image.py out some_formula.png
```

`read_image.py` is the whole recipe in about 80 lines: prepare the picture, run the encoder once, then the decoder one token at a
time with its cache. The details that matter:

* **The picture:** gray, scaled so the longer side is 448 px (the proportions are kept, not squeezed), normalised with mean
  0.9545467 and spread 0.15394445, and put in the top left corner of a 448 x 448 square that is zero elsewhere.
* **The decoder:** start token 0, end token 2, vocabulary of 15000 (`tokenizer.json` of the original repo). The first step runs
  with `use_cache_branch = false` and any zero tensors as the (unused) past; later steps run with `true`, one token, and the keys
  and values of the step before. The 24 keys/values of the attention to the picture come from the first step and stay; the 24 of
  the tokens grow.
* **Every formula comes out between `\[` and `\]`;** strip them.
* **Handwriting: draw the pen thin.** If you draw pen strokes yourself, make the line about 1 px wide at 448 px, a fiftieth of the
  height of a letter. TexTeller learned from printed formulas and reads a thin line far better than a thick one (below).

## How well it reads

Measured on [MathWriting](https://github.com/google-research/google-research/tree/master/mathwriting) test inks (handwriting from
pen strokes, drawn into pictures as above). "Exact" means the whole formula matches after ignoring spacing, braces around single
characters, `\left`/`\right` and similar spellings of the same thing.

| set                                           | exact  | within 2 characters | character error rate |
|-----------------------------------------------|--------|---------------------|----------------------|
| 1,500 inks, these 8-bit files                 | 74.1 % | 84.7 %              | 0.070                |
| the 785 of them with simple labels            | 80.8 % | 93.9 %              | 0.041                |
| first 300 inks, original float32 files        | 76.7 % | 85.3 %              | 0.071                |
| the same 300 inks, these 8-bit files          | 72.3 % | 84.0 %              | 0.078                |

So 8 bits cost about 4 points of exact matches here. The thickness of the pen matters more than that: on 100 other inks
(validation set, read with the PyTorch version of the model), a pen 2 % of a letter high read 76 % exactly right, 10 % read 39 %,
and 16 % read 25 %.

Speed (onnxruntime on a desktop CPU, one formula of a handful of tokens): about 1.35 s on one thread, 0.6 s on four. Inside a
Flutter app through the `flutter_onnxruntime` plugin: 0.56 s median per formula on four threads, once the sessions are open.

**Caveats:** MathWriting is mostly research-level maths, not school work. TexTeller's own README says the released 3.0 weights
include the test sets of the handwritten benchmarks it used and does not say which; a check of its published handwriting labels
against MathWriting's test labels found 3.8 % in common, and it read those no better than the rest, so there is no sign that
MathWriting's test inks were memorised, but this cannot be ruled out. Nothing here was measured on a phone or tablet.

## Rebuilding the files

```bash
python make_models.py out     # about 5 minutes; downloads the 1.25 GB originals (set HF_HUB_DOWNLOAD_TIMEOUT=300 on a slow line)
```

The encoder comes out byte for byte the same each time. The decoder's checksum can differ between builds (the merged graph is not
written in a fixed order), but the two builds I compared read 40 of 40 handwritten inks identically.

SHA-256 of the files in the release:

```
8ea8a061533b7fd8fbb746b0bd9ddd8d26d324fe07647dcf9059ddd07c2d77c4  encoder.onnx
ba01e6781c1acf48781e838433c185e71feccae11f40d84ee2a46ecca7ec5ad9  decoder.onnx
```

How to publish them as a GitHub release: see [RELEASING.md](RELEASING.md).

## Licence

Apache License 2.0, as TexTeller: see [LICENSE](LICENSE) and [NOTICE](NOTICE). The scripts are licensed the same way. The training
data of TexTeller comes from several sources whose own terms are not repeated here; if you ship these files widely or
commercially, look at them yourself.
