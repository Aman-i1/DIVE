"""Prose content for the generated documentation site.

Kept apart from :mod:`dive.reporting` so the HTML chrome and the words live in
different files. The CLI reference is generated from the live click command tree,
which means flags and defaults in the docs cannot drift away from the flags the
tool actually accepts.
"""

from __future__ import annotations

import html
from typing import List


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _card(title: str, body: str) -> str:
    return f'<section class="card"><h2>{_esc(title)}</h2>{body}</section>'


# ----------------------------------------------------------------------
def index_body() -> str:
    return "".join(
        [
            _card(
                "What this does",
                """
<p>DIVE trains a zoo of models on a tabular file, tunes the leaders,
stacks them, and reports the result - without writing any modelling code.
It also checks the dataset for the mistakes that silently invalidate a model:
target leakage, duplicate rows spanning the train/holdout split, and
distribution drift.</p>
<h3>Capabilities</h3>
<ul>
  <li><strong>Data profiling</strong> - problem type, class imbalance, missing
      values, high-cardinality columns, constant and ID-like columns.</li>
  <li><strong>Feature engineering</strong> - drops junk columns, expands dates
      into calendar parts, clips outliers, groups rare categories, and applies
      frequency and target encoding.</li>
  <li><strong>Model zoo</strong> - linear models, random forest, extra trees,
      histogram gradient boosting, MLP, KNN, AdaBoost, plus XGBoost, LightGBM and
      CatBoost when installed. GPU is detected automatically.</li>
  <li><strong>Tuning</strong> - Optuna search with pruning, under a wall-clock
      budget you set.</li>
  <li><strong>Stacking</strong> - out-of-fold predictions from the top models
      feed a meta-learner.</li>
  <li><strong>Crosschecks</strong> - leakage, duplicate-row leakage, drift,
      fold-to-fold stability, and a predict-time schema gate.</li>
  <li><strong>Reports</strong> - a self-contained HTML report with the
      leaderboard, diagnostics, feature importance, and every crosscheck.</li>
</ul>
""",
            ),
            _card(
                "The commands",
                """
<p>Seven commands. Every one accepts <code>--help</code> and
<code>--version</code>; full flags and defaults are in the
<a href="cli-reference.html">CLI reference</a>.</p>
<table>
<tr><th>Command</th><th>What it does</th></tr>
<tr><td><code>dive train</code></td><td>Profile, engineer features, train the zoo, tune, stack, then write the model plus a full report.</td></tr>
<tr><td><code>dive validate</code></td><td>Run the crosschecks with no training - an "is my data ready" gate.</td></tr>
<tr><td><code>dive predict</code></td><td>Score new rows with a saved model, after checking the incoming schema against training.</td></tr>
<tr><td><code>dive explain</code></td><td>Plain-English pipeline summary, top features, and code to reproduce the model.</td></tr>
<tr><td><code>dive report</code></td><td>Re-render the full HTML report from a saved model.</td></tr>
<tr><td><code>dive docs</code></td><td>Open or serve this documentation site locally.</td></tr>
<tr><td><code>dive deps</code></td><td>Show which optional packages are installed and what each one unlocks.</td></tr>
</table>
<div class="note">Every command exits <code>0</code> on success and non-zero on
failure, so <code>dive validate</code> works as a CI gate.</div>
""",
            ),
            _card(
                "Install",
                """
<h3>From a clone</h3>
<pre><code>git clone https://github.com/Aman-i1/DIVE.git
cd DIVE
pip install -e .</code></pre>
<p>That gives you the <code>dive</code> command on Windows, macOS, and Linux.
Verify with <code>dive --help</code>.</p>

<h3>Optional extras</h3>
<p>The tool runs on scikit-learn alone. Extras widen the model zoo and unlock
tuning; anything missing is skipped with a note telling you what and why.</p>
<pre><code>pip install -e ".[full]"        # everything
pip install -e ".[boosters]"    # xgboost, lightgbm, catboost
pip install -e ".[tuning]"      # optuna
pip install -e ".[explain]"     # category_encoders, shap</code></pre>
<p>Run <code>dive deps</code> at any time to see what is active.</p>
""",
            ),
            _card(
                "Google Colab",
                """
<p>No local setup. Paste this into a Colab cell:</p>
<pre><code>!git clone https://github.com/Aman-i1/DIVE.git
%cd DIVE
!pip install -q -e ".[full]"

!dive train --data examples/sample.csv --target diagnosis \\
    --mode fast --output /content/out

from IPython.display import HTML
HTML(open('/content/out/report.html').read())</code></pre>
<p>The report is a single self-contained file with the plots inlined, so it
renders directly in the notebook and can be downloaded on its own.</p>
<p>A ready-made notebook lives at <code>examples/colab_quickstart.ipynb</code>.</p>
""",
            ),
            _card(
                "Where output files land",
                """
<p>Everything from a run goes into <code>--output</code> (default
<code>./dive_output</code>):</p>
<table>
<tr><th>File</th><th>Contents</th></tr>
<tr><td><code>model.pkl</code></td><td>Fitted model plus the feature engineer, used by <code>dive predict</code>.</td></tr>
<tr><td><code>leaderboard.csv</code></td><td>Every model and its scores, best first.</td></tr>
<tr><td><code>report.html</code></td><td>Self-contained report: leaderboard, diagnostics, importance, crosschecks.</td></tr>
<tr><td><code>validation.json</code></td><td>Machine-readable crosscheck verdicts.</td></tr>
<tr><td><code>metadata.json</code></td><td>Run settings, schema, timings, skipped models.</td></tr>
<tr><td><code>plots/*.png</code></td><td>Leaderboard, train-vs-holdout, diagnostics, feature importance.</td></tr>
</table>
<div class="note">File names are stable, so scripts and CI jobs can depend on
them.</div>
""",
            ),
        ]
    )


# ----------------------------------------------------------------------
def quickstart_body() -> str:
    return "".join(
        [
            _card(
                "Five minutes, start to finish",
                """
<p>Using the bundled sample dataset - 600 rows of synthetic diagnostic data with
a deliberate ID column, a date column, and some missing values.</p>

<h3>1. Check the data before training</h3>
<pre><code>dive validate --data examples/sample.csv --target diagnosis</code></pre>
<p>Runs the crosschecks with no training. This is the step that catches a
leaking column before you spend an hour believing a 99% score.</p>

<h3>2. Train</h3>
<pre><code>dive train --data examples/sample.csv --target diagnosis \\
    --mode fast --output ./out</code></pre>
<p><code>fast</code> finishes in seconds. Use <code>balanced</code> for real work,
<code>competition</code> when you want every last point of accuracy.</p>

<h3>3. Read the report</h3>
<pre><code>dive report --model ./out/model.pkl --output ./out/report.html</code></pre>
<p>Open <code>report.html</code> in any browser.</p>

<h3>4. Score new rows</h3>
<pre><code>dive predict --model ./out/model.pkl --data new_patients.csv \\
    --output predictions.csv</code></pre>
<p>The incoming columns are checked against the training schema first. A missing
feature column stops the run instead of quietly producing wrong numbers.</p>

<h3>5. See how it works</h3>
<pre><code>dive explain --model ./out/model.pkl --output explanation.html</code></pre>
<p>A plain-English account of every pipeline stage, plus standalone Python that
rebuilds the model without dive.</p>
""",
            ),
            _card(
                "Choosing a mode",
                """
<table>
<tr><th>Mode</th><th>Models</th><th>Tuning</th><th>Stacking</th><th>Use when</th></tr>
<tr><td><code>fast</code></td><td>3 small</td><td>no</td><td>no</td>
    <td>First look at a dataset, very large data, CI smoke tests.</td></tr>
<tr><td><code>balanced</code></td><td>full zoo</td><td>yes</td><td>yes</td>
    <td>The default. Good accuracy for reasonable time.</td></tr>
<tr><td><code>competition</code></td><td>full zoo + AdaBoost</td><td>yes</td><td>yes</td>
    <td>Maximum accuracy; adds mutual-information feature selection.</td></tr>
</table>
<div class="note">Every mode respects <code>--time-budget</code> (seconds). The
run stops starting new models when the budget runs low, and reports what it
skipped rather than silently truncating.</div>
""",
            ),
            _card(
                "Using it as a library",
                """
<pre><code>import pandas as pd
from dive import Dive

df = pd.read_csv("examples/sample.csv")

dive = Dive(target="diagnosis", mode="fast", time_budget=300)
dive.fit(df)

print(dive.leaderboard())
predictions = dive.predict(df.drop(columns=["diagnosis"]))
dive.save("model.pkl")</code></pre>
<p>Reload later with <code>Dive.load("model.pkl")</code>. The fitted
feature engineer travels with the model, so <code>predict</code> never refits.</p>
""",
            ),
            _card(
                "Config files",
                """
<p>Any <code>train</code> flag can live in a YAML file:</p>
<pre><code># settings.yaml
data: examples/sample.csv
target: diagnosis
mode: balanced
time_budget: 900
output: ./out
test_size: 0.2
random_state: 42</code></pre>
<pre><code>dive train --config settings.yaml</code></pre>
<p>Explicit command-line flags override the file. An unknown key in the config
is reported as an error rather than ignored, so a typo cannot silently do
nothing.</p>
""",
            ),
        ]
    )


# ----------------------------------------------------------------------
def cli_reference_body() -> str:
    """Build the reference from the live click tree, then append the check guide."""
    blocks: List[str] = [
        _card(
            "Global usage",
            """
<pre><code>dive [--quiet] [--traceback] COMMAND [OPTIONS]</code></pre>
<table>
<tr><th>Flag</th><th>Effect</th></tr>
<tr><td><code>--version</code>, <code>-V</code></td><td>Print the version and exit.</td></tr>
<tr><td><code>--help</code>, <code>-h</code></td><td>Available on every command.</td></tr>
<tr><td><code>--quiet</code>, <code>-q</code></td><td>Suppress progress output; warnings and errors still print.</td></tr>
<tr><td><code>--traceback</code></td><td>Print the full Python traceback on failure.</td></tr>
</table>
<p>Exit codes: <code>0</code> success, <code>1</code> a user or data error
(including a failed validation), <code>2</code> an unexpected internal error.</p>
""",
        )
    ]
    blocks.append(_generated_command_reference())
    blocks.append(_crosscheck_guide())
    return "".join(blocks)


def _generated_command_reference() -> str:
    """Render every command's real options straight out of click."""
    import click

    from dive.cli import cli

    context = click.Context(cli, info_name="dive")
    parts: List[str] = []

    for name in sorted(cli.list_commands(context)):
        command = cli.get_command(context, name)
        if command is None or getattr(command, "hidden", False):
            continue

        help_text = (command.help or "").strip()
        summary, _, remainder = help_text.partition("\n")
        rows: List[str] = []
        for parameter in command.params:
            if not isinstance(parameter, click.Option):
                continue
            flags = ", ".join(f"<code>{_esc(o)}</code>" for o in parameter.opts)
            required = " <strong>(required)</strong>" if parameter.required else ""
            description = _esc(parameter.help or "")
            rows.append(f"<tr><td>{flags}{required}</td><td>{description}</td></tr>")

        example = _EXAMPLES.get(name, "")
        example_html = f"<pre><code>{_esc(example)}</code></pre>" if example else ""
        remainder_html = (
            f"<p>{_esc(remainder.strip())}</p>" if remainder.strip() else ""
        )
        table = (
            "<table><tr><th>Option</th><th>Description</th></tr>"
            + "".join(rows)
            + "</table>"
            if rows
            else "<p>No options.</p>"
        )
        parts.append(
            _card(
                f"dive {name}",
                f"<p>{_esc(summary)}</p>{remainder_html}{table}{example_html}",
            )
        )
    return "".join(parts)


_EXAMPLES = {
    "train": "dive train --data sales.csv --target revenue --mode balanced \\\n"
             "    --time-budget 900 --output ./out",
    "predict": "dive predict --model ./out/model.pkl --data new_rows.csv \\\n"
               "    --proba --output scored.csv",
    "validate": "dive validate --data sales.csv --target revenue",
    "explain": "dive explain --model ./out/model.pkl --output explanation.html",
    "report": "dive report --model ./out/model.pkl --output report.html",
    "docs": "dive docs            # open the docs in your browser\n"
            "dive docs --serve    # serve them on http://localhost:8000",
    "deps": "dive deps",
}


def _crosscheck_guide() -> str:
    return _card(
        "What each crosscheck means",
        """
<p><code>dive validate</code> and the pre-flight stage of <code>dive train</code>
run the same suite. Each check reports <span class="verdict pass">PASS</span>,
<span class="verdict warn">WARN</span>, or <span class="verdict fail">FAIL</span>.</p>

<h3>target_health</h3>
<p>Class imbalance, a near-constant target, or heavy target missingness.</p>
<ul>
  <li><strong>WARN</strong> - imbalance beyond 10:1, or over 20% of targets missing.
      Accuracy becomes misleading; read balanced accuracy and F1 instead.</li>
  <li><strong>FAIL</strong> - one class holds 99%+ of rows, the target barely varies,
      or the smallest class has under 5 rows. There is nothing learnable here.</li>
</ul>

<h3>target_leakage</h3>
<p>Scores every feature against the target: absolute correlation for numeric
pairs, normalised mutual information otherwise.</p>
<ul>
  <li><strong>WARN</strong> - a feature scores 0.90-0.98. Strong, possibly legitimate.
      Worth understanding before you trust it.</li>
  <li><strong>FAIL</strong> - a feature scores 0.98 or above. That is a re-encoding of
      the answer, not a predictor. Drop it; otherwise the model looks excellent in
      testing and fails in production.</li>
</ul>

<h3>duplicate_rows</h3>
<p>Identical feature rows, and specifically rows appearing on both sides of the
train/holdout split.</p>
<ul>
  <li><strong>WARN</strong> - duplicates exist within the dataset. They bias
      frequency encodings and inflate scores.</li>
  <li><strong>FAIL</strong> - 1% or more of holdout rows also appear in training.
      The holdout score is partly a memory test. Deduplicate, or split so
      identical records stay on one side.</li>
</ul>

<h3>train_holdout_drift</h3>
<p>Per-feature Population Stability Index plus a Kolmogorov-Smirnov test between
the two splits. Both must agree before a feature is flagged.</p>
<ul>
  <li><strong>WARN</strong> - a numeric feature is distributed differently across the
      split. Expected with <code>--time-series</code>. On a random split it means
      the data is ordered or grouped, so the holdout score may not describe
      future rows.</li>
</ul>

<h3>missing_data</h3>
<ul>
  <li><strong>WARN</strong> - a column is over 40% missing, so imputation is supplying
      most of its values.</li>
  <li><strong>FAIL</strong> - a column is 99%+ missing and carries no information.</li>
</ul>

<h3>cv_stability</h3>
<p>Per-fold scores for the winning model, not just the mean.</p>
<ul>
  <li><strong>WARN</strong> - the standard deviation exceeds 10% of the mean. The
      headline number depends on which rows landed in which fold; expect real
      performance anywhere in the fold range.</li>
  <li><strong>SKIP</strong> - boosted models and stacked ensembles are scored on the
      holdout rather than by cross-validation, so no fold scores exist.</li>
</ul>

<h3>predict_schema</h3>
<p>Runs inside <code>dive predict</code>, comparing incoming columns and dtypes
against training.</p>
<ul>
  <li><strong>WARN</strong> - a column changed type. The pipeline usually copes;
      check the values are sensible.</li>
  <li><strong>FAIL</strong> - a required column is missing. Scoring stops rather than
      misaligning columns and returning confident nonsense. Columns dropped during
      training (constants, IDs) are allowed to be absent, and extra columns are
      ignored.</li>
</ul>

<div class="note"><strong>Exit codes:</strong> <code>dive validate</code> returns
1 on any FAIL, and 0 when only warnings are present - unless you pass
<code>--strict</code>, which makes warnings fail too. Useful as a CI gate.</div>
""",
    )
