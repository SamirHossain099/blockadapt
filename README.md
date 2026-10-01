# blockadapt

Online adaptation of vision-aided millimetre-wave blockage prediction under deployment shift, evaluated as
streams on DeepSense 6G scenarios 17-21.

A camera at the receiver of a fixed 60 GHz link predicts, from the last eight frames, whether the line of sight
will be blocked within the next second. A predictor trained on one afternoon loses accuracy at night and in other
recording sessions. This repository compares two ways of adapting it online, without any human label:

- six label-free test-time adaptation methods (normalisation statistics, Tent, EATA, SAR, CoTTA, T3A);
- delayed self-labels: one horizon later the link's own received power says whether a blockage followed, and
  the predictor is fine-tuned online on that, or only its decision threshold is re-calibrated.

It also holds the rule that derives the blockage state from received power and its agreement with the dataset's
labels, a shuffled-batch control that separates stream correlation from the shift, and the average precision
along each stream.

## Layout

- `src/blk/`: `data.py` (domains, horizon labels, the power rule), `model.py` (the predictor), `tta.py` (online
  methods, `step` / `feedback` protocol), `metrics.py`
- `scripts/`: `cache_frames.py` (frames, labels, power and time stamps of a scenario into one array file),
  `run.py` (source training and the streams), `label_agreement.py` (domain table, power rule against the labels),
  `headline.py` and `stream_curve.py` (aggregation), `manuscript_numbers.py`
- `results/`: aggregated JSON outputs; `figures/`: figures at 600 dpi and as PDF; `tests/`: every number
  quoted in the paper is read from `results/` by a test

## Data

DeepSense 6G scenarios 17 to 21 (https://deepsense6g.net), licence as stated by the dataset; not redistributed,
and neither are the frame caches or the trained models. Set the extracted root with the environment variable
`DEEPSENSE_ROOT`.

Two things about the data that the code handles and a user should know:

- The recorded frame rates differ from the nominal ones: the index files' time stamps give about 12 frames/s
  on scenarios 17 and 18, 12 falling to 7 within scenario 19, 6 on scenario 20 and 5 on scenario 21. The
  prediction horizon is set in seconds and converted per sequence.
- In scenario 20 the index file names the label files of its last sequence by the `index` column, which jumps by
  6,584 at that sequence, while the label files are numbered by row. `cache_frames.py` reads them by row; the
  check against received power is in `results/label_agreement.json`.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/cache_frames.py --scenarios scenario17 scenario18 scenario19 scenario20 scenario21
python scripts/label_agreement.py
python scripts/run.py --seeds 0 1 2 3 4 --orders sequential iid --tag _F5
python scripts/headline.py
python scripts/stream_curve.py
python scripts/manuscript_numbers.py
python -m pytest tests -q
python src/figures.py
```

`run.py` needs a GPU with about 8 GB of memory; the five targets and five seeds take several hours. It writes
the per-frame scores that `stream_curve.py` reads to `results/scores/`, which is not part of this repository.

## Citation

See `CITATION.cff`. Archived at Zenodo, concept DOI 10.5281/zenodo.23072763 (always the latest release). Cite DeepSense 6G (Alkhateeb et al., IEEE Communications Magazine, 2023) and the
vision-aided blockage prediction study that introduced these scenarios (Charan and Alkhateeb, IEEE Globecom
Workshops, 2022) for the data.

## Licence

MIT.
