# Kikuyu test clips vs African Next Voices (ANV) train and validation

Source: public copy `badrex/anv_data_ke_kikuyu_scripted` (metadata columns only). Test set: 300 clips, 23 speakers.

| ANV part | rows | speakers | our speakers found | our clips by those speakers | our sentences found |
|---|--:|--:|--:|--:|--:|
| train | 111197 | 408 | 0/23 | 0/300 | 61/300 |
| validation | 13346 | 48 | 23/23 | 300/300 | 300/300 |

Split labels in the copy: test/dev_test 6522, train/train 111197, validation/dev 6824, validation/dev_test 6522.
The copy's validation files contain the dev_test rows again, so validation must never be used for training or model selection here.

61 test clips have a sentence that also appears in ANV train (read by other speakers). Report Kikuyu results on all clips and on the remaining unseen-sentence clips, and leave these sentences out of any language-model or fine-tuning text.
