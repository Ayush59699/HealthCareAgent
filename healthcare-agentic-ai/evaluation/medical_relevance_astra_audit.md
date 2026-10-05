# Audit of the 92 Astra relevance proposals

**Recommendation: change 7 labels; retain 85.** All 92 remain model-assisted proposals with `reviewed: false`. This is a second-pass model audit, not independent human review or clinical advice. Neither input file was modified.

The audit uses the label-free findings and every entire stored passage. It excludes final diagnosis, titles, retrieval/lexical ranks, cosine similarity and pooling metadata. Missing passage context and unlisted patient findings were not filled in.

## Exact label changes

All seven revised labels have **medium** confidence. Match on the full chunk ID, not the ordinal or a rank. The JSON audit contains replacement reasoning, confidence and difficult-interpretation notes in each change's `set` object, plus `expected_before` preconditions.

| Chunk ID | Passage content | Before | Apply |
|---|---|---|---|
| `14150db3-ef7b-50df-a72c-602fbc19ad4f` | Breast pain and local breast warning signs | `partially_relevant` | **`relevant`** |
| `1bdacc9b-3025-5941-9ecd-55fc7858edb0` | Diverticular abdominal cramping and pain pattern | `not_relevant` | **`partially_relevant`** |
| `2c804e1f-2a3e-5e60-a147-99bab21e2909` | Palpitations definition in an unattributed symptom fragment | `uncertain` | **`partially_relevant`** |
| `534fa256-b2fc-5594-a8bd-e3326224127a` | Cardiac chest/upper-abdominal pain and dyspnea in a women-focused passage | `partially_relevant` | **`relevant`** |
| `62f37f8c-4548-59fb-937c-6664715c0f10` | Heart-attack chest/stomach pain, dyspnea and variable evolution | `partially_relevant` | **`relevant`** |
| `ab5c1cb6-2012-5b10-b4dd-cb4818d362a9` | Self-contained abdominal-pain warning signs, disease omitted | `uncertain` | **`partially_relevant`** |
| `f8da5cff-193e-5a8e-991a-fe46e53364e8` | Coronary chest pain/dyspnea and activity/rest pattern | `partially_relevant` | **`relevant`** |

### Why these changes

- **`14150db3-ef7b-50df-a72c-602fbc19ad4f`:** The original label discounted directly applicable local symptom information mainly because cancer is unusual at this age. The benchmark asks about useful information for one or more findings, not the probability of the named disease. The same local symptom content is treated as relevant in the general breast-disease passage.
- **`1bdacc9b-3025-5941-9ecd-55fc7858edb0`:** The original rationale treated the typical lower-abdominal site as an exclusive location and overlooked the broader abdominal-pain and onset information. Under the existing partial-relevance rubric used for other GI comparisons, this is a limited mismatch, not a complete absence of useful content.
- **`2c804e1f-2a3e-5e60-a147-99bab21e2909`:** An unidentified disease prevents a disease-specific judgment but does not erase the self-contained definition of an observed symptom. The benchmark permits usefulness for a finding without requiring attribution to a named disease. Do not import the missing disease from a title or another candidate.
- **`534fa256-b2fc-5594-a8bd-e3326224127a`:** The original rationale used pediatric applicability to demote a substantive cardiopulmonary symptom comparison while retaining the general heart-attack passage as relevant. Apply the same relevance-versus-likelihood distinction to both; preserve the age/sex limitation in reasoning and confidence rather than suppressing useful content.
- **`62f37f8c-4548-59fb-937c-6664715c0f10`:** Unlike an unattributed symptom fragment, this passage names its condition and retains a coherent symptom/time-course explanation. Lack of pediatric risk information is a limitation, not a reason to rate this direct comparison below the related general heart-attack passage solely on age.
- **`ab5c1cb6-2012-5b10-b4dd-cb4818d362a9`:** The original uncertainty conflated inability to name the disease with inability to identify any useful information. The warning-sign relation is preserved in the text, so partial relevance can be assigned while explicitly retaining the missing-context and symptom-severity limitations.
- **`f8da5cff-193e-5a8e-991a-fe46e53364e8`:** The original downgrade mixed disease probability with passage relevance and gave undue weight to nonradiating pain, although radiation is optional in the text. Retain age and risk caveats while treating this substantive cardiac symptom comparison consistently with other relevant cardiac passages.

## Label totals

| Label | Before | After |
|---|---:|---:|
| `relevant` | 25 | 29 |
| `partially_relevant` | 38 | 37 |
| `not_relevant` | 24 | 23 |
| `uncertain` | 5 | 3 |

## Retain these three as uncertain

- **`30f0143f-4f36-5471-9fd7-bce57d3ed1b3`:** Retain uncertain: the orphan list loses both the topic and the relationship of asthma/depression/pain to that topic; words alone cannot reconstruct a risk or medication relationship.
- **`ab272626-6a5b-5c57-a447-7be1847efd0a`:** Retain uncertain: the list names neither its syndrome nor a self-contained clinical relationship/definition; anxiety/cramps alone cannot establish its intended meaning.
- **`cb0e6ef6-19f7-5515-b366-857516e9620b`:** Retain uncertain: the opening chest-pain/dyspnea clause has an unknown subject; TSS begins a new proposition and cannot automatically supply that subject.

The distinction is not simply whether a fragment names a disease. The palpitations fragment preserves a definition, and the abdominal-pain fragment preserves an explicit warning-sign relationship. The remaining uncertain fragments lack the subject or relationship needed to interpret the overlapping terms reliably.

## Important retained-label decisions

- **Keep direct panic/anxiety, asthma and cardiopulmonary comparisons relevant.** This is retrieval relevance, not selection of a diagnosis. Do not demote rare alternatives solely because of age or promote psychiatric content solely because of an assumed final diagnosis.
- **Keep generic risk/management background partial.** The existing partial rubric allows limited contextual usefulness without treating management information as episode-level diagnostic evidence.
- **Keep leukemia-only night-sweat lists not relevant.** Unqualified increased sweating is not a documented nocturnal pattern. ALL/CLL passages with explicit below-rib pain remain partial for that separate anatomical overlap, not because leukemia is suspected.
- **Keep PMS and ectopic-pregnancy symptom content conditional/partial.** Menstrual development, cyclicity and pregnancy context are unknown. Do not infer pregnancy, exposure or impossibility from age alone.
- **Keep unsupported postpartum, menopausal, neonatal and unrelated anatomical contexts not relevant.** A generic shared word is not enough to supply the missing clinical context.
- **Keep neurologic distinctions explicit.** Focal arm compression has only partial overlap with bilateral limbs/perioral tingling; a sensory/motor/autonomic explanation is directly useful without proving neuropathy.

## How to apply without corrupting review provenance

1. Verify the two input hashes below.
2. Use `medical_relevance_astra_audit.json` as the complete audited review aid, or make a **new copy** of the original proposals.
3. In that new copy, match each `changes[].chunk_id`, check `expected_before`, and apply `set`. The remaining 85 candidate labels stay unchanged.
4. Update aggregate counts and uncertain/difficult-case lists from the audit summary; do not retain stale uncertainty lists from the original proposals.
5. Keep every `reviewed` value false. Do not overwrite the original benchmark/proposals or treat this audit as human-reviewed gold labels.

- Benchmark SHA-256: `173cb7199f9403e299208ecd7a6ff06a47b54d451a04383bf78c852171032cfc`
- Original proposals SHA-256: `a9db1d0b3dc3c5398fbf390d40166fabf195c6a20491898b704c055c74f88f9e`

## Complete 92-candidate label set

The ordinal below is source-file enumeration only, not retrieval rank. Full replacement rationale and per-record audit reasoning are in the JSON artifact.

| Record | Chunk ID | Recommended proposed label | Confidence | Action |
|---:|---|---|---|---|
| 1 | `02831ea8-2625-5518-87b0-855aebd2235e` | `partially_relevant` | medium | retain |
| 2 | `038699b9-dc86-5e12-be0c-b413e3aaea98` | `relevant` | medium | retain |
| 3 | `0463c267-1066-58c1-8f38-2be3b83e74ae` | `partially_relevant` | medium | retain |
| 4 | `091e945b-9959-5ba6-a340-52f1a9a7abdc` | `partially_relevant` | medium | retain |
| 5 | `0be3b274-83b9-5388-9052-e6f1ea5c691d` | `not_relevant` | medium | retain |
| 6 | `10426618-94dc-597e-85bf-36f733c4a0cd` | `partially_relevant` | high | retain |
| 7 | `10632734-8cbf-5bfc-8ef7-4df95552969f` | `relevant` | high | retain |
| 8 | `1221c8b1-1f77-5da7-9fa1-87e89f28d916` | `not_relevant` | medium | retain |
| 9 | `14150db3-ef7b-50df-a72c-602fbc19ad4f` | `relevant` | medium | change |
| 10 | `15aebb38-be4c-5e63-83b7-6667367c6cb9` | `not_relevant` | medium | retain |
| 11 | `177b755e-067b-5991-b67b-2beb99e8ccb2` | `relevant` | high | retain |
| 12 | `198af62a-531a-56a4-af8f-17f0f600700c` | `partially_relevant` | medium | retain |
| 13 | `1b8b0fb4-f510-57bf-b664-3d9b984d6826` | `relevant` | high | retain |
| 14 | `1bdacc9b-3025-5941-9ecd-55fc7858edb0` | `partially_relevant` | medium | change |
| 15 | `1d8e23e3-a321-5f70-b47d-37ff401b9a1b` | `not_relevant` | high | retain |
| 16 | `21cc511f-2980-5496-a531-b9d1f60bbc0d` | `not_relevant` | high | retain |
| 17 | `2af489a6-0e40-53d6-b4c3-0602e7eba118` | `partially_relevant` | medium | retain |
| 18 | `2c804e1f-2a3e-5e60-a147-99bab21e2909` | `partially_relevant` | medium | change |
| 19 | `2f71f19a-c9ed-5856-a388-14f532ca71b6` | `relevant` | high | retain |
| 20 | `2fbf87ea-2c76-50d0-b97f-4024eacd1ab3` | `partially_relevant` | medium | retain |
| 21 | `30f0143f-4f36-5471-9fd7-bce57d3ed1b3` | `uncertain` | low | retain |
| 22 | `346c5001-9afa-54b5-aa2d-e2bb6681e799` | `partially_relevant` | medium | retain |
| 23 | `39d51780-d727-5895-8231-a1cf04c46f95` | `not_relevant` | high | retain |
| 24 | `3b3d7cab-7175-5537-b044-67fba3c17259` | `partially_relevant` | high | retain |
| 25 | `3ca61f7f-5c0d-59cf-8ed3-bae57470e837` | `relevant` | high | retain |
| 26 | `3cd34cc2-1fa2-5ddc-8b24-8e206119fa04` | `not_relevant` | medium | retain |
| 27 | `3d401cef-def7-5be4-8d91-c81dc37be215` | `not_relevant` | high | retain |
| 28 | `4081c3f4-8a43-553e-9ada-b91c853cfb32` | `partially_relevant` | high | retain |
| 29 | `4afdc8a3-e816-57dd-aabb-f9d243987da2` | `partially_relevant` | medium | retain |
| 30 | `4bcfbb4d-9b1e-570f-9a77-b86f12cc02d3` | `not_relevant` | high | retain |
| 31 | `4fa2f16e-8f86-5d41-b7ec-a66a36b68493` | `partially_relevant` | medium | retain |
| 32 | `534fa256-b2fc-5594-a8bd-e3326224127a` | `relevant` | medium | change |
| 33 | `553fb5f7-0eaa-58b7-983d-8aa2174ac374` | `not_relevant` | high | retain |
| 34 | `58a8e084-1882-557e-8aec-6fb5719992bb` | `relevant` | high | retain |
| 35 | `5e5c0617-d136-5bdd-a41f-f02edd17403d` | `relevant` | high | retain |
| 36 | `62f37f8c-4548-59fb-937c-6664715c0f10` | `relevant` | medium | change |
| 37 | `693ea7c9-fd2a-5e22-945f-bd9c7c97ef6b` | `partially_relevant` | medium | retain |
| 38 | `6a35be9a-42ab-5eba-8d1f-63cce708795e` | `partially_relevant` | high | retain |
| 39 | `6c1ab54d-8296-5b9b-b542-369b64565015` | `partially_relevant` | medium | retain |
| 40 | `6d26b866-f904-5fa8-b24a-c28e28a58274` | `not_relevant` | high | retain |
| 41 | `72fa5e8b-af32-55b0-a830-9bb092ca9b3c` | `relevant` | high | retain |
| 42 | `73521ab3-c2df-5d05-adcf-7e327b934d85` | `relevant` | high | retain |
| 43 | `76cb8951-ff4a-5827-ad51-dd69ef332cee` | `partially_relevant` | high | retain |
| 44 | `76fbcc71-565f-50f1-b1c3-193a935b7371` | `relevant` | high | retain |
| 45 | `7711d0fa-1b36-5c16-b1ce-f77ab647a478` | `not_relevant` | high | retain |
| 46 | `7745c91d-6ae8-5137-85a6-df42528aa91f` | `not_relevant` | high | retain |
| 47 | `7767017c-fd10-5a17-91f4-d12a6370b0d5` | `partially_relevant` | medium | retain |
| 48 | `77cf9c04-c634-575e-81bb-3f8f711c6f45` | `not_relevant` | high | retain |
| 49 | `7f4a6aae-57a0-5480-9fa4-8820b2e76a29` | `not_relevant` | high | retain |
| 50 | `81b21366-e0cb-5300-ade5-13e0c3e7c33b` | `not_relevant` | medium | retain |
| 51 | `831a8efb-e442-5d96-8ddd-033063bfc4b4` | `partially_relevant` | medium | retain |
| 52 | `83c518cf-fe61-59bd-a952-2b7961071313` | `partially_relevant` | medium | retain |
| 53 | `8bf19141-28e2-52ac-a948-1c5b646c2759` | `relevant` | medium | retain |
| 54 | `8d3b0e86-a4e9-52b2-aac3-f464d82536e6` | `partially_relevant` | medium | retain |
| 55 | `91f2a8f0-f316-5c90-8d94-73f21579d805` | `partially_relevant` | medium | retain |
| 56 | `9218536a-a168-5067-8bcd-c46275353714` | `partially_relevant` | medium | retain |
| 57 | `92ebd8da-30d6-5bed-aaa4-482ad6804acc` | `relevant` | high | retain |
| 58 | `963ee5e9-6fea-5cf1-afdc-df7ef2560204` | `partially_relevant` | medium | retain |
| 59 | `990b4be4-8a79-5357-b0b9-abdca86647d0` | `not_relevant` | high | retain |
| 60 | `9cba8661-2749-5a7a-8155-b123a82a121b` | `partially_relevant` | medium | retain |
| 61 | `9d4a774f-9e67-5ad6-9868-663acab9b05d` | `relevant` | high | retain |
| 62 | `9da28175-590a-52bf-acf1-a43d1e0f2f99` | `relevant` | high | retain |
| 63 | `a780ed9f-7047-5424-927a-7460814e476a` | `not_relevant` | high | retain |
| 64 | `ab272626-6a5b-5c57-a447-7be1847efd0a` | `uncertain` | low | retain |
| 65 | `ab5c1cb6-2012-5b10-b4dd-cb4818d362a9` | `partially_relevant` | medium | change |
| 66 | `af149c15-5718-5ca4-9f68-62914ce69039` | `partially_relevant` | medium | retain |
| 67 | `afbf0d1a-e66b-5496-a7c9-d68437640c94` | `partially_relevant` | medium | retain |
| 68 | `b1193978-a8b0-507e-8616-547347be35e6` | `relevant` | high | retain |
| 69 | `b6c83f1c-f3a2-52de-aa6a-41eb814dbf79` | `relevant` | high | retain |
| 70 | `b748bd5a-530a-5998-bc3f-881319f72027` | `not_relevant` | high | retain |
| 71 | `b82d53fc-13e4-5a45-82ed-663c911304b5` | `partially_relevant` | high | retain |
| 72 | `bb6ddda2-ff89-5c93-b400-1fa33efce687` | `partially_relevant` | medium | retain |
| 73 | `c4e0833d-84ae-5830-8848-bc3dd9cdaff8` | `relevant` | medium | retain |
| 74 | `c6be0fe7-daa6-5ab8-bcce-23205dc8a76f` | `relevant` | medium | retain |
| 75 | `c6c80e91-fe4e-5e8e-bcaa-97981d8c5394` | `relevant` | high | retain |
| 76 | `c8474a93-5486-5066-af19-25df5e3d246f` | `relevant` | high | retain |
| 77 | `cacb3894-7cab-589a-824c-cf7abca2b4b6` | `relevant` | high | retain |
| 78 | `cb0e6ef6-19f7-5515-b366-857516e9620b` | `uncertain` | low | retain |
| 79 | `ce21184b-190d-5e2f-aa03-14612cb7477f` | `partially_relevant` | medium | retain |
| 80 | `d1c9e558-ca3b-53ea-b09e-5e05ef7671b0` | `relevant` | high | retain |
| 81 | `d2e1db2a-60c9-5eba-b24d-3ad2183a8f3d` | `relevant` | high | retain |
| 82 | `d3173238-6590-50ab-acab-670c4f02b86c` | `not_relevant` | medium | retain |
| 83 | `d5f29d05-b8b9-5c64-a18b-86913bc08804` | `not_relevant` | high | retain |
| 84 | `d7ec5dc7-4b3b-5eb1-bd31-38e04fbf07ff` | `not_relevant` | medium | retain |
| 85 | `db5fe0bb-6475-5e72-927d-d49b5bece4e4` | `not_relevant` | high | retain |
| 86 | `e1b5acd7-cbbd-5861-b57d-4746194ca0ad` | `partially_relevant` | medium | retain |
| 87 | `e7b041fd-76dc-5ebf-8302-a2687b0fc5d3` | `partially_relevant` | medium | retain |
| 88 | `e859d1ae-f9ba-582f-8c05-d4e01e1e1294` | `partially_relevant` | medium | retain |
| 89 | `f75b1529-602a-59f7-84c6-ad4982014249` | `partially_relevant` | high | retain |
| 90 | `f8da5cff-193e-5a8e-991a-fe46e53364e8` | `relevant` | medium | change |
| 91 | `fb9cd89f-828e-5b1d-9e64-b784e71648eb` | `relevant` | high | retain |
| 92 | `fe9616b9-5de2-5d50-817f-02b56c2da92a` | `partially_relevant` | medium | retain |

## Medically difficult interpretations

26 candidates are explicitly flagged. A difficult interpretation can still have a usable relevance label; it is not automatically uncertain.

- **`038699b9-dc86-5e12-be0c-b413e3aaea98`** (`relevant`): A rare neurovisceral disorder overlaps several findings, but episode duration, triggers and objective testing are unknown; relevance is distinct from diagnostic likelihood.
- **`14150db3-ef7b-50df-a72c-602fbc19ad4f`** (`relevant`): Distinguish useful local breast symptom characterization from the very different question of pediatric cancer probability.
- **`1bdacc9b-3025-5941-9ecd-55fc7858edb0`** (`partially_relevant`): Typical lower-left pain differs from the observed upper-abdominal/flank pain, but the text does not make the typical location obligatory; this is a borderline partial/not-relevant decision.
- **`2c804e1f-2a3e-5e60-a147-99bab21e2909`** (`partially_relevant`): The palpitations definition survives truncation, but disease attribution and sex-specific applicability remain unknown.
- **`2fbf87ea-2c76-50d0-b97f-4024eacd1ab3`** (`partially_relevant`): Menstrual development and cyclic timing are unknown; PMS cannot be presumed or categorically excluded from age alone.
- **`30f0143f-4f36-5471-9fd7-bce57d3ed1b3`** (`uncertain`): The fragment loses both its topic and the relationship of asthma/depression/pain to the remaining list.
- **`346c5001-9afa-54b5-aa2d-e2bb6681e799`** (`partially_relevant`): Fear of dying during an episode is not equivalent to repeated thoughts of death or self-harm in a child mental-health warning list.
- **`4afdc8a3-e816-57dd-aabb-f9d243987da2`** (`partially_relevant`): The absolute statement that congenital heart defects do not cause pain needs clinical caution; its dyspnea content remains of limited differential use.
- **`534fa256-b2fc-5594-a8bd-e3326224127a`** (`relevant`): Direct cardiac symptom comparison remains useful, but adult sex-specific epidemiology and disease probability must not be transferred to a child.
- **`62f37f8c-4548-59fb-937c-6664715c0f10`** (`relevant`): The fragment retains its cardiac context; that makes its symptom content interpretable despite truncation and the rarity of pediatric ischemia.
- **`693ea7c9-fd2a-5e22-945f-bd9c7c97ef6b`** (`partially_relevant`): A postmenopausal statement has no preserved subject and must be separated from the intact variant-angina explanation.
- **`7767017c-fd10-5a17-91f4-d12a6370b0d5`** (`partially_relevant`): Recorded head trauma is not interchangeable with psychological trauma, and the passage describes a later usual age of onset.
- **`8bf19141-28e2-52ac-a948-1c5b646c2759`** (`relevant`): The symptom cluster overlaps strongly, but ischemic heart attack is an unusual pediatric explanation; confidence concerns relevance, not diagnosis.
- **`8d3b0e86-a4e9-52b2-aac3-f464d82536e6`** (`partially_relevant`): PMS treatment discussion is both management-only and conditional on unestablished menstrual/cyclic context.
- **`9218536a-a168-5067-8bcd-c46275353714`** (`partially_relevant`): A fragment about thoughts of death and injury must not be mapped to suicidal intent from the reported fear of dying.
- **`963ee5e9-6fea-5cf1-afdc-df7ef2560204`** (`partially_relevant`): The passage is truncated and mainly describes an adult-only asthma procedure, limiting its pediatric usefulness.
- **`9cba8661-2749-5a7a-8155-b123a82a121b`** (`partially_relevant`): Isolated abdominal pain overlaps, but ovarian-cancer interpretation is highly age- and examination-dependent.
- **`ab272626-6a5b-5c57-a447-7be1847efd0a`** (`uncertain`): The condition is unnamed, and anxiety/cramps overlap while suicide-related language risks a misleading interpretation.
- **`ab5c1cb6-2012-5b10-b4dd-cb4818d362a9`** (`partially_relevant`): Disease context and applicability of the full warning cluster are unknown, but the explicit warning-sign relation is preserved; do not infer tenderness or severe illness from the pain rating.
- **`af149c15-5718-5ca4-9f68-62914ce69039`** (`partially_relevant`): Pregnancy-related differential content is conditional on unknown reproductive history and development; neither pregnancy nor impossibility is inferred.
- **`c4e0833d-84ae-5830-8848-bc3dd9cdaff8`** (`relevant`): Prodromal symptom overlap does not establish current cardiac arrest or predict impending collapse.
- **`cb0e6ef6-19f7-5515-b366-857516e9620b`** (`uncertain`): The condition attached to chest pain/dyspnea is missing; the subsequent TSS paragraph must not be used to fill that gap.
- **`e1b5acd7-cbbd-5861-b57d-4746194ca0ad`** (`partially_relevant`): Focal upper-limb nerve compression is a limited match to bilateral limb and perioral paresthesias, with an additional age mismatch.
- **`e7b041fd-76dc-5ebf-8302-a2687b0fc5d3`** (`partially_relevant`): Below-rib pain overlaps, but CLL has especially limited applicability in a child and nonspecific sweating is not drenching night sweats.
- **`f8da5cff-193e-5a8e-991a-fe46e53364e8`** (`relevant`): Distinguish direct cardiac symptom-pattern information from pediatric atherosclerotic risk; optional radiation and unknown activity triggers are not required findings.
- **`fe9616b9-5de2-5d50-817f-02b56c2da92a`** (`partially_relevant`): Prior depression and family psychiatric history do not establish bipolarity; fear of dying is not evidence of suicidal thoughts.
