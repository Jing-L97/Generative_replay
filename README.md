# kNN-LMs and the lexical frequency gap in syntactic contrasts

## Setup

```bash
pip install -r requirements.txt
pip install faiss-cpu hydra-core   # not pinned in requirements.txt yet
```

On macOS, `faiss-cpu` and PyTorch ship separate OpenMP runtimes. The smoke-test
script sets `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` to keep both
loadable in the same process.

## Layout

```
src/knn_lm/
├── datastore/            FAISS index + (key, value) memmaps
│   ├── capturer.py         · ActivationCapturer hook on final-FFN input
│   ├── build_index.py      · IVFPQ (default) / IndexFlatL2 (tiny_mode)
│   ├── build_full.py       · Full contextualized datastore (KNNSaver)
│   └── build_semantic.py   · Averaged + GloVe/fastText variants (§2.3)
│
├── retrieval/
│   ├── knn_wrapper.py            · Eqs. 1–4 from the manuscript, λ=0.25
│   ├── structural_similarity.py  · SV / Wh / RC pattern detectors
│   └── structural_filter.py      · w, wo, w+r, wo+r conditions (Fig. 4)
│
├── preprocess/
│   └── corpus.py         · Chunk corpus into token / phrase / sequence units
│
└── eval/
    ├── loaders.py           · BLiMP / Zorro / BIG-Bench → canonical JSONL
    ├── frequency_strata.py  · High (>10⁴) vs low (<10³) stratification
    └── minimal_pair.py      · P(grammatical) > P(ungrammatical) scoring

src/scripts/
├── datastore/
│   ├── build_full.py        · Hydra CLI for the full datastore
│   └── build_semantic.py    · Hydra CLI for semantic datastores
└── eval/
    └── eval_minimal_pairs.py  · End-to-end: build datastore + retrieve + score

experiments/configs/        · Hydra config groups (model, datastore,
│                            retrieval, eval) and smoke-test compositions
└── tests/                  · Smoke fixtures + test_smoke.sh
```

To run a single config:

```bash
python -m scripts.eval.eval_minimal_pairs \
    --config-dir experiments/configs/tests \
    --config-name tiny_full
```

To sweep the structural conditions:

```bash
for c in w wo w+r wo+r; do
  python -m scripts.eval.eval_minimal_pairs \
      --config-dir experiments/configs/tests \
      --config-name tiny_structural \
      retrieval.condition=$c \
      output.path=results/smoke/structural_$c.json
done
```

## Running the manuscript experiments

The smoke configs are self-contained. For the real experiments, compose Hydra
config groups under `experiments/configs/`:

```bash
# Example: full datastore, sequence-level, k=16, τ=10
python -m scripts.eval.eval_minimal_pairs \
    --config-dir experiments/configs \
    +model=gpt2_xl \
    +datastore=full \
    +retrieval=sequence \
    +eval=blimp \
    retrieval.k=16 retrieval.tau=10 \
    corpus.path=/path/to/your/babylm_train.txt \
    eval.freq_path=/path/to/word_freq.json \
    output.path=results/gpt2xl_full_seq_k16.json
```
