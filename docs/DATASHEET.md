# TrustNet Dataset Datasheet

Following the standard dataset reporting framework proposed by Gebru et al. (2021) (*Datasheets for Datasets*, Communications of the ACM).

---

## 1. Motivation
- **For what purpose was the dataset created?**  
  To train, evaluate, and calibrate multimodal fact-checking models across textual claims, contextual imagery, retrieved evidence documents, and multilingual Indian language contexts (Hindi, Hinglish, English).
- **Who created the dataset and on behalf of which entity?**  
  The TrustNet research and development team, integrating open-access benchmark corpora with verified public fact-checks.
- **Who funded the creation of the dataset?**  
  Open research initiative for civil society information integrity.

---

## 2. Composition
- **What do the instances that comprise the dataset represent?**  
  Each record represents a single verifiable claim paired with optional digital imagery, associated evidence passage IDs from indexed corpus documents, gold verification verdicts (`verified`, `contradicted`, `insufficient_evidence`), and language metadata.
- **How many instances are there in total?**  
  Partitioned into five immutable splits tracked in `data/manifest.json`:
  - `train_text`: 5 instances (text model fine-tuning)
  - `train_fusion`: 3 instances (multimodal fusion training)
  - `val_select`: 2 instances (hyperparameter tuning)
  - `val_cal`: 2 instances (temperature calibration and abstention estimation)
  - `val`: 4 instances (validation benchmarking)
  - `test`: 3 instances (final held-out evaluation split, SHA-256: `c5ed7665f11e555139f96916541ab5310faee03ac7747098883223b9548387c2`)
- **Does the dataset contain all possible instances or is it a sample?**  
  It is a curated evaluation sample designed to test edge cases, cross-lingual consistency, and epistemic gaps.
- **What data does each instance consist of?**  
  - `id`: Unique identifier (string).
  - `claim`: Claim text as forwarded or asserted (string).
  - `verdict`: Ground-truth verdict (string: `verified`, `contradicted`, `insufficient_evidence`).
  - `evidence_doc_ids`: List of document identifiers in the reference evidence corpus.
  - `image_path`: Optional local filesystem path to associated image.
  - `language`: Language code (`en`, `hi`, `hinglish`).

---

## 3. Collection Process
- **What mechanisms or procedures were used to collect the data?**  
  Collection combined claims from FEVER, Snopes fact-check archives, and Indian fact-checking platforms (BOOM Live, Alt News archives, PIB Fact Check public releases).
- **Who was involved in the data collection process?**  
  NLP researchers and fact-checking annotators.
- **Over what timeframe was the data collected?**  
  Historical fact-checking archives spanning 2020 through 2026.

---

## 4. Preprocessing, Cleaning, and Labeling
- **Was any preprocessing/cleaning done?**  
  - Unicode NFKC normalization and Devanagari script standardization.
  - Hinglish Roman-to-Devanagari phonological transliteration via dictionary and rule-based pipelines.
  - Exif metadata sanitization to protect user privacy while extracting timestamp provenance.
- **How were ground-truth labels assigned?**  
  Labels were mapped to a standard 3-way taxonomy (`verified`, `contradicted`, `insufficient_evidence`) adhering to FEVER / PolitiFact guidelines.

---

## 5. Uses
- **Has the dataset been used for tasks previously?**  
  Yes, across TrustNet Stages 1 through 11 for retrieval indexing, multimodal feature fusion, temperature scaling, and REST API testing.
- **What tasks should the dataset NOT be used for?**  
  - Training general-purpose conversational LLMs without factuality grounding.
  - Automated social credit scoring or individual content filtering.

---

## 6. Distribution and Maintenance
- **How will the dataset be distributed?**  
  Stored locally within the repository under `data/processed/` with strict cryptographic digests verified by `data/manifest.json`.
- **Who supports and maintains the dataset?**  
  TrustNet Open Source Maintainers.
