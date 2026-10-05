# CHECK3: opt-in WHO LLM catalog selector (selector-only)

## Outcome

Implemented a separate catalog-selector experiment. **Production WHO lookup, AMG, Diagnostic, Grounding, Critic, Safety and all existing runtime code were unchanged during this task.** Their pre-existing uncommitted changes from earlier tasks were preserved, not overwritten.

One live selector request saw the complete **569-document metadata catalog**, not document bodies. It selected five valid IDs. Only those five local documents were subsequently loaded, sequentially. **No live Diagnostic/Critic/Safety pipeline was run.**

The two previous `known`/`chronic` false matches disappeared from this run. This does **not** prove general negation understanding, optimal selection, clinical relevance of all five selections, or improved diagnosis.

## Selected documents

| Returned ID | Title | Type | Local filename |
|---|---|---|---|
| `w:975ab3bb8f7c` | Anaemia | Fact Sheet | `anaemia.txt` |
| `w:2637bb7299d1` | Menstrual health | Fact Sheet | `menstrual-health.txt` |
| `w:05100ecb40ef` | Malnutrition | Fact Sheet | `malnutrition.txt` |
| `w:27ff7ac1294d` | Diabetes | Fact Sheet | `diabetes.txt` |
| `w:86629bd046c4` | Hypertension | Fact Sheet | `hypertension.txt` |

All are existing files in `data/who_fact_sheets/`. Both Fact Sheets and Q&A were available in the catalog; the model chose no Q&A on this call. No document was downloaded, synthesized, altered, indexed or duplicated into a new corpus.

### Old lexical versus new selection

Same patient, with the exact prior label-free payload and negations retained:

| Old lexical result (existing max 3) | Literal matched word | New selector result |
|---|---|---|
| Headache disorders: How common are headaches? | `headaches` | Not selected |
| Radiation: The known health effects of ultraviolet radiation | `known` | Not selected |
| Chronic obstructive pulmonary disease (COPD) | `chronic` | Not selected |

The latter two words came from **“No known chronic illness.”** Neither of those two titles was selected by the LLM. The model instead returned the five titles above. Old lexical results were obtained with the unchanged query constructor and `WHOLookup.search(..., max_results=3)` (metadata only). They were **not shown to the selector**.

The comparison preserves the old three-result setting versus the requested new maximum of five. It is not a controlled equal-size precision/recall benchmark, and no relevance labels or expected diagnosis were used.

### Are these the right documents?

A bounded inspection after selection found that the Anaemia source explicitly discusses fatigue, reduced physical capacity and dizziness/light-headedness. That makes it a plausible informational match to the vignette, **not confirmation that the patient has anaemia**.

The other selections need scrutiny:

- Menstrual health discusses bleeding-related issues, but menstrual/bleeding history was not supplied.
- Malnutrition is broad; no dietary or weight history was supplied.
- Diabetes mentions tiredness, but the case supplies no glucose results or characteristic metabolic history.
- Hypertension discusses headache/dizziness in some contexts, but no blood pressure was supplied. Standing-associated dizziness does not establish hypertension.

The selector was forbidden to output diagnoses or explanations; its accepted output contains **only selected IDs**. Consequently, we do not claim to know its justification or that it inferred any of those conditions. Selecting the maximum five documents once also does not demonstrate careful selectivity. The observed removal of the two generic-word matches is encouraging but insufficient to promote this approach.

## Implementation

New files only (apart from required OKF synchronization):

- `rag/who_catalog_selector.py`: bounded metadata catalog, strict IDs-only response contract, validation, and sequential local selected-document loader.
- `scripts/evaluate_who_catalog_selector.py`: explicit opt-in selector-only CLI and old/new comparison.
- `tests/test_who_catalog_selector.py`: 24 offline contract tests.
- This report; artifacts under the gitignored `outputs/check3-selector/` directory.

Architecture implemented in this phase:

```text
label-free manual patient payload + WHO metadata catalog
    -> existing cloud provider, selector-specific instructions
    -> {selected_ids: [...]} (0–5 unique known IDs)
    -> validate every ID before opening any document
    -> sequential local .txt reads, source-header checks, SHA-256 provenance
    -> selector-only audit/report
    -> STOP (no Diagnostic/Critic/Safety invocation)
```

The catalog includes only **ID, complete title, type and filename**, encoded as a columnar table to avoid redundant JSON keys. IDs are deterministic metadata hashes with collision detection; a full catalog SHA-256 binds the mapping. All 569 entries were included, without truncation or lexical prefiltering.

There is no public catalog-enumeration method in the existing lookup. The isolated adapter reads `WHOLookup._entries`, which contains validated frozen metadata. It explicitly fails on index errors, incompatible entry types, ambiguous filenames, collisions or catalog limits. This private-API dependency is documented rather than adding an API to production WHO lookup.

Strict output checks reject unknown IDs, duplicates, more than five IDs, wrong types, and extra fields such as diagnoses or explanations. Empty selection is valid. Failed generation never triggers document loading. All IDs are revalidated at the loading boundary, even for a caller bypassing the normal provider validator. There is no fallback search or document substitution.

Selected files retain title, URL, topic, type, filename, retrieval timestamp, full-document SHA-256, length, catalog fingerprint and truthful `selection_method=who-catalog-selector-v1`. Original WHO indexes do not contain document checksums; SHA-256 is calculated from the loaded source bytes, using the same convention as existing WHO provenance. Independent verification also matched every selected checksum against the **pre-run source hashes**.

The existing WHO grounding payload specifically declares lexical matching and a lexical body-prefix selection method. **LLM selection is not falsely relabeled as a lexical match.** This phase exposes full selected source text through its own typed `LoadedWHO` iterator, but intentionally does not connect it to the Diagnostic snapshot. Any later downstream integration requires separate authorization and truthful provenance compatibility; it was not attempted here.

### Negations and scope

The selector receives the exact original `patient.to_inference_dict()` with “No fever,” “No chest pain,” “No shortness of breath,” “No known chronic illness,” “No current medications,” the three-week duration, and unavailable laboratory information intact. No LLM patient rewrite is performed. The selector instructions explicitly distinguish negative/unknown statements from positive findings, disallow demographic-only assumptions, and require IDs only.

No prior Diagnostic output, expected answer, benchmark label or hand-picked disease name was sent as a query hint. Disease titles are present only as entries in the requested WHO catalog. The instruction examples about `known`, `chronic` and negations come from this task and its patient wording, not a hidden diagnosis.

The original `prompts/check3.txt` patient vignette was replaced by the current task instructions. For this run, the patient was restored **only from `outputs/check3/live/input.json`**, not from a diagnosis file. Its content-bound manual ID matched the original saved case:

`manual:c4fde5e8d602403b0a565bb23518fbaeab41d39217827f4a147847ed52564524`

### Memory / request bounds

- At most 1000 catalog entries and 90000 bytes of metadata payload; exceedance fails rather than truncating.
- Actual catalog: 569 entries; complete provider request: **69103 bytes**, within the existing 100000-byte cap.
- Zero document bodies sent to the selector; no local embedding models or vector stores initialized.
- At most five selected files; existing 4 MiB per-document bound; yielded sequentially with no text cache. The runner retains only audit metadata, not a concatenated full-document collection.
- One synchronous cloud request, no retry, 512 output-token cap. Existing provider/config files are unchanged; these are local experimental settings only.
- No downloads, database/index builds or parallel cloud calls.

## Validation and tests

Live request: **4.26 seconds**, **18662 input tokens**, **62 output tokens**, one successful call, no retries or truncation. The deployment's default sampling behavior was retained; the API does not receive a temperature setting. This is a single stochastic observation, not a reproducibility guarantee.

Checks independently verified:

- Patient payload exactly equals the original saved label-free input.
- Request contains metadata catalog and patient facts only.
- Returned IDs are unique, within the catalog, and within the five-document bound.
- Returned metadata matches the catalog entries.
- All five loaded source checksums match both pre-run and current local files.
- No downstream pipeline was run.

Tests:

- **24 new selector tests passed.** Metadata-only catalog construction, complete field inventory, stable IDs, source loading/checksums, negation preservation, label-bearing input rejection, empty selection, five-ID cap, duplicate/unknown ID rejection, forged objects, diagnoses/extra-field rejection, provider failure, context budget refusal, file/header/path failures, catalog limits/collisions, saved-case restoration, and a mocked selector-only CLI run.
- Selector + existing WHO lookup + manual-input focused suite: **61 passed**.
- Full offline project regression: **707 run, 706 passed, one existing opt-in cloud skip**. These include mocked workflow tests, not a new live downstream run. The same already-installed Python package-path workaround documented in the previous CHECK3 report was used; no dependencies were installed or modified.
- `git diff --check` passed.

Source-integrity inventory: 820 pre-existing files, covering Python sources, WHO documents/indexes, manifests and the current task file. Only the two OKF manifests change in this task; **all inventoried pre-existing production/experiment Python code and WHO assets remain byte-identical**. Earlier dirty working-tree changes remain present but are not new changes made by this task. See `outputs/check3-selector/integrity.json`.

## Reproduce

From `healthcare-agentic-ai`:

```text
# Prepare catalog and lexical baseline only; zero cloud calls
python scripts/evaluate_who_catalog_selector.py --saved-case outputs/check3/live/input.json --output-dir outputs/check3-selector/my-preview

# Selector-only live call; output directory must be new
python scripts/evaluate_who_catalog_selector.py --saved-case outputs/check3/live/input.json --live --output-dir outputs/check3-selector/my-live

# Alternatively supply a sectioned manual vignette, not the task instruction file
python scripts/evaluate_who_catalog_selector.py --case-file path/to/vignette.txt --live

python -m unittest tests.test_who_catalog_selector tests.test_who_lookup tests.test_manual_patient
```

Saved artifacts: `outputs/check3-selector/live/report.json`, `selector_request.json`, `verification.json`; preview and test logs in the parent directory. No selected full-text documents were copied into these artifacts.

## Recommendation

Keep this isolated and selector-only. The requested `known/chronic` false matches are absent in the observed selection, and Anaemia is a plausible informational improvement. However, the other four selections do not yet demonstrate sound case-specific selection. Review relevance and over-selection before authorizing downstream integration; do not declare improved diagnosis, change production lookup, or run the Diagnostic/Critic/Safety pipeline based on this one result.
