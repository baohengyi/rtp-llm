# Diagnostic latest-main integration candidate

NOT READY FOR DEVELOPMENT-BRANCH MERGE OR FULL ACCEPTANCE.

This commit snapshots the pending Python Native integration for early CI feedback. No golden, comparator or tolerance changes are authorized by this snapshot. Legacy test sources and 26 unresolved Python methods are retained; preserving source does not mean those contracts compile or execute on the new cache/tracing APIs.

Outstanding: 215 ordinary legacy cache declarations, 31 behavior-contract decisions (5 are included in the 215), 9 fixed layout skips out of 46 registrations, all backend execution and wheel checks. Do not infer acceptance from targeted CI.

Initial scope: CUDA native library/test compilation and frontend 73-case execution. No H20/GB200 GPU test execution is requested by that compilation scope.
