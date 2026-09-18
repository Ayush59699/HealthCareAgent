# Dataset location

The five original DDXPlus release files are already available at
`../../../data/ddxplus/` relative to this directory. They are not copied or modified.
The parser uses that location automatically unless release metadata exists here.
Use `--data-dir` or `DDXPLUS_DATA_DIR` to select another location.

You may place your authorized copies of release_evidences.json,
release_conditions.json, release_train_patients.zip,
release_validate_patients.zip and release_test_patients.zip here.
No extraction is required. Do not commit datasets or generated patient exports.
