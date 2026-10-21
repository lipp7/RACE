# Third-party components

The project-level MIT license covers the RACE code. Third-party components keep
their own copyright and license notices.

- `third_party/PromptSRC/` contains the MIT-licensed CLIP/tokenizer modules,
  BPE vocabulary, MaPLe prompt definitions, and template definitions from the
  pinned PromptSRC fork. The upstream `LICENSE` is retained. The default CLIP
  download location is changed to the project-relative `cache/clip`; model and
  prompt equations are unchanged.
- `third_party/MaPLe/` retains the original MaPLe trainer definition, five-epoch
  configuration, and MIT license. This allows the wrapper to verify that the
  prompt definition agrees with the pinned fork.
- CoOp, CoCoOp, LaBo, Tip-Adapter-F, ProKeR, and ProLIP in `maft/` and `scripts/`
  are explicit implementations/ports of the corresponding methods for the
  matched experimental protocol. Citations and upstream identities are recorded
  in their source comments and `third_party/versions.json`.
- The LaBo candidate bank is downloaded separately from its pinned upstream
  repository by `race.fetch_labo`. Its images, weights, and candidate bank are
  not part of this distribution.

OpenAI/OpenCLIP pretrained weights are downloaded separately into the local
cache. Image datasets are obtained separately from their maintainers.
