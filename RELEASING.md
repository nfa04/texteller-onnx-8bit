# Publishing the two files as a release

The files are too big for a normal git push (GitHub refuses single files over 100 MB; `decoder.onnx` is 229 MB). Put them in a
release instead (up to 2 GB per file). The repository has to be public for a program to download them without logging in.

```bash
python make_models.py out          # about 5 minutes; downloads 1.25 GB of originals first
cat out/SHA256SUMS                 # put these two lines in README.md (the table at the end)
gh release create v1 out/encoder.onnx out/decoder.onnx out/SHA256SUMS \
   --title "TexTeller 3.0, 8-bit ONNX" \
   --notes "8-bit conversion of TexTeller 3.0 (Apache-2.0). See README.md for what was changed and how to use it."
```

Anyone can then download them from

    https://github.com/<you>/<repo>/releases/download/v1/encoder.onnx
    https://github.com/<you>/<repo>/releases/download/v1/decoder.onnx

For UniNotes, build or run the app with

    --dart-define=TEXTELLER_REPO=https://github.com/<you>/<repo>/releases/download/v1

and the **Download** button in Settings → Formulas fetches both files. (GitHub redirects these links to its file servers; the
downloader follows redirects.)

A new release for changed files gets a new tag (`v2`): the app builds that point at `v1` keep working.
