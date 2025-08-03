# Helpful General Instructions

To get HDBSCAN added to scikit-learn, the submitter undertook a structured, disciplined process. Here’s a comprehensive guide, mapping each concrete step with the analogous actions Micky774 (the submitter of the referenced PR) followed. This workflow is **highly relevant to anyone adding a new research algorithm to scikit-learn**.

## 1. **Research and Prototype Development**

- **Understand the Algorithm:** Start with a thorough literature review to fully understand the algorithm (HDBSCAN in this case). Test prototypes in standalone scripts or notebooks using a separate package if available (the hdbscan package existed prior to integration)[1](https://github.com/scikit-learn-contrib/hdbscan)[2](https://www.geeksforgeeks.org/machine-learning/hdbscan/).
- **Analogous Action:** Micky774 and others have long used the `hdbscan` Python library, which behaves similarly to other `sklearn.cluster` estimators, ensuring a clear idea of both expected functionality and API consistency[1](https://github.com/scikit-learn-contrib/hdbscan).

## 2. **Preliminary Discussion with Community**

- **Open Issue or Proposal:** Early interaction with scikit-learn maintainers is crucial, either by opening a GitHub issue or a draft PR describing motivation, use-cases, and initial approach.
- **Why:** This avoids duplicating work, ensures alignment with maintainers' expectations, and clarifies design choices[3](https://scikit-learn.org/stable/developers/contributing.html)[4](https://scikit-learn.org/0.21/developers/contributing.html).

## 3. **Fork and Prepare the Codebase**

- **Fork and Clone:** Fork the main scikit-learn repository and clone it locally.
- **Setup Environment:** Ensure the development environment matches scikit-learn’s requirements (Python version, packages, etc.).
- **Analogous Action:** The submitter forked and worked from their own branch (`hdbscan_sync` in the PR)[5](https://github.com/scikit-learn/scikit-learn/pull/26385).

## 4. **Implement the Estimator**

- **API Consistency:** Build the estimator as a class complying with scikit-learn's estimator API:
    - Correct method signatures (`fit`, `predict`, `fit_predict`, `get_params`, `set_params`)
    - Hyperparameters set as keyword arguments in `__init__` with defaults
    - No logic in `__init__` (just attribute assignment)[4](https://scikit-learn.org/0.21/developers/contributing.html)[6](https://scikit-learn.org/stable/developers/develop.html)
    - Complete docstrings, including parameters and examples
- **Use Internal Structures:** For performance, parts of HDBSCAN were written in Cython (`_tree.pxd`), closely tracking how scikit-learn’s clustering modules implement hierarchical structures.

## 5. **Testing and Validation**

- **Unit Tests:** Put tests under `sklearn/tests` or within the module directory. Use `pytest` and scikit-learn’s utilities like `check_estimator`.
- **Examples:** Add practical examples of usage (and possibly comparisons with similar estimators) in the examples gallery[3](https://scikit-learn.org/stable/developers/contributing.html).
- **Analogous Action:** The PR added Cython code, updated internal structures, and provided robust unit tests as well as computational benchmarks[5](https://github.com/scikit-learn/scikit-learn/pull/26385).

## 6. **Documentation**

- **User Guide Integration:** Add detailed class and function-level docstrings. Update the User Guide with relevant theory, usage, and examples (plots, notebooks)[3](https://scikit-learn.org/stable/developers/contributing.html).
- **API Reference:** Ensure `sklearn.cluster.HDBSCAN` appears in the generated API docs[7](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html).

## 7. **Adhere to Contribution Guidelines**

- **Code Quality:** Match scikit-learn’s formatting, style, and best practices (`flake8`, no unrelated changes, meaningful comments)[3](https://scikit-learn.org/stable/developers/contributing.html)[4](https://scikit-learn.org/0.21/developers/contributing.html).
- **Changelog:** Add notes in the changelog summarizing the user-facing changes[3](https://scikit-learn.org/stable/developers/contributing.html).
- **Analogous Action:** The PR discussion shows attention to code structure, type annotations, and community feedback resolution[5](https://github.com/scikit-learn/scikit-learn/pull/26385).

## 8. **Pull Request Submission**

- **Open PR:** From the dedicated feature branch. Clearly summarize changes, mention any linked issues, and outline algorithm benefits.
- **Respond to Reviews:** Address feedback promptly—whether it’s about code style, API decisions, documentation, or tests.
- **Why:** Thorough review improves maintainability and maximizes the new algorithm’s usability for the whole community[8](https://blog.scikit-learn.org/community/pull-request/).
- **Analogous Action:** Micky774’s PR underwent multiple review cycles, with core maintainers and community signaling [resolved] as issues were fixed[5](https://github.com/scikit-learn/scikit-learn/pull/26385).

## 9. **Final Checks & Merge**

- **Tests and CI:** All tests must pass on continuous integration servers (Linux, Windows, macOS, etc.).
- **Merge:** After reviews are resolved and approvals received, a core maintainer merges the contribution.
- **Analogous Action:** The PR (#26385) was merged after passing CI and code review, becoming part of the main branch[5](https://github.com/scikit-learn/scikit-learn/pull/26385).

## **Summary Table: Steps from Research to scikit-learn PR Merge**

| Step | What You Do | Micky774’s PR Actions |
| --- | --- | --- |
| Prototype & research | Understand, prototype externally | Used hdbscan Python lib, API consistency |
| Pre-proposal discussion | Open issue/draft, align with maintainers | Early communication via PR |
| Fork and dev setup | Fork, clone, configure dev environment | Developed in feature branch |
| Estimator implementation | Follow scikit-learn estimator API, Cython for heavy computations if needed | Implemented HDBSCAN in estimator, Cython |
| Testing | Add comprehensive unit tests, doc tests, estimator checks | Tests, benchmarks, estimator checks |
| Documentation | Extensive docstrings, user guide, gallery examples | Updated docs/examples |
| Meet contribution standards | Lint, changelog, structured code/comments | Resolved comments, followed guidelines |
| Submit/review PR | Open PR, respond to reviewer feedback | PR revised, reviewer dialogue |
| CI & Merge | Pass CI, merge after approval | Merged May 31, 2023 after CI passed |

Each step is necessary to guarantee **code quality**, **API consistency**, and **long-term maintainability**—and to maximize community benefit from the new contribution[3](https://scikit-learn.org/stable/developers/contributing.html)[8](https://blog.scikit-learn.org/community/pull-request/)[6](https://scikit-learn.org/stable/developers/develop.html)[4](https://scikit-learn.org/0.21/developers/contributing.html)[5](https://github.com/scikit-learn/scikit-learn/pull/26385).