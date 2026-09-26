# SADU sample diagrams

A 10-image sample from the SADU benchmark ("Benchmarking and Evaluating VLMs
for Software Architecture Diagram Understanding"), used as real-world raster
architecture diagrams for `ImageConnector` fixture-level testing and the
Phase 5 caption-quality/latency spike (see `docs/ROADMAP.md`'s Multi-modal
entry).

- **Source**: Zenodo record [10.5281/zenodo.19339991](https://doi.org/10.5281/zenodo.19339991),
  file `SADU_ASE_submission.zip` (MD5 `e1d920fd43b36442427c8c36a2e7582e`).
- **License**: CC-BY-4.0.
- **Selection**: 4 real Azure architecture diagrams (`azure-openai-gateway-*`,
  `concurrent-pattern.png`, `enterprise-integration-message-broker-events.png`,
  `automate-document-classification-durable-functions.png`), 3 UML use-case
  diagrams (`g02_uc.png`, `g04-UC.png`, `US2_DC.png`), and 3 entity-relationship
  diagrams (`ER_diagram_{1,2,3}.png`) — chosen for smaller file size and
  diversity across the benchmark's three diagram categories (behavior,
  structural, ER).
- **QA files**: `ER_diagram_{1,2,3}_QA.json` are the benchmark's own
  ground-truth question/answer pairs for those three ER diagrams (e.g. exact
  entity/relation counts) — usable as an objective cross-check for caption
  quality, since a free-text caption can be sanity-checked against a known
  entity count even though it isn't asked to restate it.

Not used for anything beyond local testing/spikes — these are real,
third-party, CC-BY-4.0-licensed images, not synthetic fixtures.
