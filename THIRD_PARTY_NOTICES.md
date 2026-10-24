# Third-party notices

QWENgine contains code derived from the following projects. Their licenses are reproduced below.

| Component | Files in this repository | Origin | License |
|---|---|---|---|
| ggml quantization formats | `src/quant/quant_formats.h` (block layouts), `src/quant/iq_tables.inc` (lookup tables, extracted verbatim), `src/quant/quant_ref.cpp` (reference (de)quantizers, ported 1:1), and the device decoders built on them in `src/cuda/` and `src/cpu/` | [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp), `ggml/src/ggml-common.h`, `ggml/src/ggml-quants.c` | MIT |
| Frokenizer pre-tokenizer | `third_party/frokenizer` (git submodule; the build uses its re2c-generated `include/frokenizer_generated/lexer.hpp`) | [yassa9/frokenizer](https://github.com/yassa9/frokenizer), Copyright (c) 2026 Yassa Sfen | MIT |
| Qwen chat template | `src/server/chat.cpp` (C++ port of the Jinja template stored in the GGUF) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Apache-2.0 |
| stb_image | `include/third_party/stb_image.h` (v2.30, vendored unmodified, as in llama.cpp's `vendor/stb`) | [nothings/stb](https://github.com/nothings/stb), Copyright (c) 2017 Sean Barrett | MIT or public domain (the file's choice); MIT below |
| Pillow bicubic resampling | `src/runtime/vision.cpp` (`resize_bicubic_pillow`: port of the algorithm of `src/libImaging/Resample.c`, checked byte for byte against Pillow by `tests/unit/test_image_resize.py`) | [Pillow](https://github.com/python-pillow/Pillow) | HPND (MIT-CMU) |
| Qwen2-VL image preprocessing | `src/runtime/vision.cpp` (`VisionModel::target_size`: the smart resize; normalization) | [Hugging Face transformers](https://github.com/huggingface/transformers), `Qwen2VLImageProcessor` | Apache-2.0 |

The model files (`Qwen3.8-27B-GSQ-RCO-IQ3_XXS*.gguf` and the vision encoder `mmproj-Qwen3.8-27B-BF16.gguf`, from
[ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF): Qwen/Qwen3.8-27B
quantized by ISTA DASLab, Apache-2.0) are not distributed with this repository.

The tests and some tools link against a separate llama.cpp build when one is configured (`Q38_GGML_DIR`); no
llama.cpp binary is distributed here.

## ggml (MIT)

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Frokenizer (MIT)

From <https://github.com/yassa9/frokenizer> (`LICENSE`, also in the submodule):

```
MIT License

Copyright (c) 2026 Yassa Sfen.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Qwen chat template (Apache-2.0)

The chat template is part of the Qwen3.8-27B model release, licensed under the Apache License, Version 2.0
(<https://www.apache.org/licenses/LICENSE-2.0>). `src/server/chat.cpp` reimplements its logic in C++; its output is
checked byte for byte against the original template rendered by jinja2 (`tests/unit/test_chat.cpp`).

The Qwen2-VL image preprocessing (`smart_resize` and the normalization constants of `Qwen2VLImageProcessor`) is
reimplemented from Hugging Face transformers, also under the Apache License, Version 2.0.

## stb_image (MIT)

```
MIT License
Copyright (c) 2017 Sean Barrett
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Pillow (HPND)

```
Copyright (c) 1997-2011 Secret Labs AB
Copyright (c) 1995-2011 Fredrik Lundh
Copyright (c) 2010-2020 Alex Clark and contributors

By obtaining, using, and/or copying this software and/or its associated
documentation, you agree that you have read, understood, and will comply
with the following terms and conditions:

Permission to use, copy, modify, and distribute this software and its
associated documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appears in all copies, and that
both that copyright notice and this permission notice appear in supporting
documentation, and that the name of Secret Labs AB or the author not be
used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
```
