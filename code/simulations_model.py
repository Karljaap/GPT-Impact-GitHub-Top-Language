# -*- coding: utf-8 -*-
"""
simulations_model.py
--------------------
Reproducible simulations for the theoretical model in Appendix B.

The script generates:
  1. Structural mechanism figures: task exposure, threshold entry, covariance.
  2. Monte Carlo diagnostics for a fixed-effects panel estimator.
  3. LaTeX tables and a paper-ready LaTeX section.

Run from the project root:
  python code/simulations_model.py
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, "output", "figures", "simulations")
TAB_DIR = os.path.join(BASE_DIR, "output", "tables", "simulations")

LANGUAGES = [
    "C", "C#", "C++", "Go", "Java", "JavaScript",
    "PHP", "Python", "Ruby", "TypeScript",
]

SEED = 20260608


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def latex_escape(value):
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def dataframe_to_latex_table(df, caption, label, path, float_format="{:.3f}"):
    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{" + "l" * len(df.columns) + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(latex_escape(c) for c in df.columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        vals = []
        for val in row:
            if isinstance(val, (float, np.floating)):
                vals.append(float_format.format(float(val)))
            else:
                vals.append(latex_escape(val))
        lines.append(" & ".join(vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


@dataclass
class TaskMetrics:
    language: str
    tau: float
    rho: float
    hbar: float
    cov_norm: float
    theta: float
    exact_gain_norm: float
    auto_tasks: int


def simulate_tasks_for_language(rng, language, n_tasks, corr, advantage_mean, h_mean):
    """Simulate task-level human productivity and ChatGPT advantage."""
    z_h = rng.normal(size=n_tasks)
    eps = rng.normal(size=n_tasks)
    z_g = corr * z_h + np.sqrt(max(1.0 - corr ** 2, 0.0)) * eps

    log_h = h_mean + 0.50 * z_h
    h = np.exp(log_h)

    # a/h = exp(g), so ChatGPT is more productive when g > 0.
    g = advantage_mean + 0.65 * z_g
    a = h * np.exp(g)

    auto = a > h
    tau = float(auto.mean())
    if auto.sum() == 0:
        return TaskMetrics(language, 0.0, 0.0, np.nan, 0.0, 0.0, 0.0, 0)

    s = (a[auto] - h[auto]) / h[auto]
    h_auto = h[auto]
    hbar = float(h_auto.mean())
    rho = float(s.mean())

    # Normalize human productivity in automated tasks so that hbar = 1.
    h_norm = h_auto / hbar
    cov_norm = float(np.mean((h_norm - 1.0) * (s - rho)))
    theta = tau * rho
    exact_gain_norm = tau * (rho + cov_norm)
    return TaskMetrics(language, tau, rho, hbar, cov_norm, theta, exact_gain_norm, int(auto.sum()))


def simulate_language_exposure(seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    means = np.linspace(-0.28, 0.22, len(LANGUAGES))
    h_means = np.linspace(-0.10, 0.18, len(LANGUAGES))
    rng.shuffle(means)
    rng.shuffle(h_means)

    for lang, m, hm in zip(LANGUAGES, means, h_means):
        metric = simulate_tasks_for_language(
            rng=rng,
            language=lang,
            n_tasks=12000,
            corr=0.0,
            advantage_mean=float(m),
            h_mean=float(hm),
        )
        rows.append(metric.__dict__)

    df = pd.DataFrame(rows).sort_values("theta", ascending=False).reset_index(drop=True)
    df["approx_error_pct"] = 100.0 * (df["exact_gain_norm"] - df["theta"]) / df["exact_gain_norm"]
    return df


def plot_language_exposure(exposure_df):
    plot_df = exposure_df.sort_values("theta", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(plot_df["language"], plot_df["tau"], color="#8ecae6", label=r"$\tau_c$: automated-task share")
    ax.scatter(plot_df["theta"], plot_df["language"], color="#b00020", s=45, label=r"$\theta_c=\tau_c\rho_c$")
    ax.set_xlabel("Task share / exposure index")
    ax.set_ylabel("Language")
    ax.set_title("Simulated task exposure by programming language")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    savefig(os.path.join(FIG_DIR, "sim_language_exposure.png"))


def plot_threshold_entry():
    x = np.linspace(-3.5, 3.5, 900)
    old_threshold = 0.70
    delta_eta = 0.55
    new_threshold = old_threshold - delta_eta
    density = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(x, density, color="#12355b", lw=2.3)
    ax.fill_between(
        x,
        0,
        density,
        where=(x >= new_threshold) & (x < old_threshold),
        color="#ffb703",
        alpha=0.65,
        label="Marginal entrants",
    )
    ax.axvline(old_threshold, color="#b00020", lw=2, label=r"Old threshold $\eta_c^*(0)$")
    ax.axvline(new_threshold, color="#008000", lw=2, label=r"New threshold $\eta_c^*(1)$")
    ax.annotate(
        r"$\Delta\eta=\tau_c\rho_c$",
        xy=((old_threshold + new_threshold) / 2, 0.18),
        xytext=(-1.55, 0.27),
        arrowprops=dict(arrowstyle="->", color="#333333"),
        fontsize=12,
    )
    ax.set_xlabel(r"Individual programming ability $\eta$")
    ax.set_ylabel("Density")
    ax.set_title("Lower entry threshold induced by AI productivity gains")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    savefig(os.path.join(FIG_DIR, "sim_threshold_entry.png"))


def simulate_country_language_entry(exposure_df, seed=SEED + 1):
    rng = np.random.default_rng(seed)
    n_countries = 60
    countries = [f"C{i:02d}" for i in range(1, n_countries + 1)]
    phi = rng.lognormal(mean=0.0, sigma=0.35, size=n_countries)
    phi = phi / phi.mean()

    rows = []
    for country, phi_i in zip(countries, phi):
        for _, lang_row in exposure_df.iterrows():
            theta = float(lang_row["theta"])
            rows.append({
                "country": country,
                "language": lang_row["language"],
                "phi_i": phi_i,
                "theta_c": theta,
                "delta_y_index": phi_i * theta,
            })
    return pd.DataFrame(rows)


def plot_theta_entry_relationship(entry_df):
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    sc = ax.scatter(
        entry_df["theta_c"],
        entry_df["delta_y_index"],
        c=entry_df["phi_i"],
        cmap="viridis",
        s=30,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.25,
    )
    ax.set_xlabel(r"Language exposure $\theta_c=\tau_c\rho_c$")
    ax.set_ylabel(r"Predicted entry index $\phi_i\theta_c$")
    ax.set_title("Exposure and absorption jointly determine predicted entry")
    ax.grid(alpha=0.2)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"Absorption index $\phi_i$")
    savefig(os.path.join(FIG_DIR, "sim_theta_entry_relationship.png"))


def covariance_sensitivity(seed=SEED + 2):
    rng = np.random.default_rng(seed)
    corr_grid = np.linspace(-0.8, 0.8, 17)
    rows = []
    for corr in corr_grid:
        metric = simulate_tasks_for_language(
            rng=rng,
            language="Representative",
            n_tasks=60000,
            corr=float(corr),
            advantage_mean=0.02,
            h_mean=0.0,
        )
        rows.append({
            "corr_input": corr,
            "tau": metric.tau,
            "rho": metric.rho,
            "cov_norm": metric.cov_norm,
            "theta_simplified": metric.theta,
            "exact_gain_norm": metric.exact_gain_norm,
            "relative_gap_pct": 100.0 * (metric.exact_gain_norm - metric.theta) / metric.exact_gain_norm,
        })
    return pd.DataFrame(rows)


def plot_covariance_sensitivity(cov_df):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(cov_df["corr_input"], cov_df["exact_gain_norm"], marker="o",
            color="#12355b", label="Exact normalized gain")
    ax.plot(cov_df["corr_input"], cov_df["theta_simplified"], marker="s",
            color="#b00020", label=r"Simplified gain $\tau_c\rho_c$")
    ax.axvline(0, color="#444444", lw=1, ls="--")
    ax.set_xlabel(r"Correlation between $\log h_c(z)$ and AI advantage")
    ax.set_ylabel("Gain index")
    ax.set_title("Sensitivity to the covariance assumption")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    savefig(os.path.join(FIG_DIR, "sim_covariance_sensitivity.png"))


def build_panel_design(exposure_df, seed=SEED + 3, n_countries=50, n_quarters=16):
    rng = np.random.default_rng(seed)
    languages = exposure_df["language"].tolist()
    n_languages = len(languages)
    countries = [f"C{i:02d}" for i in range(1, n_countries + 1)]
    quarters = np.arange(1, n_quarters + 1)

    phi = rng.lognormal(mean=0.0, sigma=0.35, size=n_countries)
    phi = phi / phi.mean()
    theta = exposure_df.set_index("language")["theta"].to_dict()
    theta_mean = np.mean(list(theta.values()))
    theta = {k: v / theta_mean for k, v in theta.items()}

    treated = rng.binomial(1, 0.5, size=n_countries)
    alpha = rng.normal(0.0, 1.0, size=n_countries)
    beta_t = 0.05 * (quarters - quarters.mean()) + rng.normal(0.0, 0.05, size=n_quarters)
    delta = rng.normal(0.0, 0.45, size=n_languages)

    rows = []
    for ci, country in enumerate(countries):
        for li, language in enumerate(languages):
            for ti, q in enumerate(quarters):
                post = int(q >= 12)
                a_it = int(treated[ci] == 1 and post == 1)
                x = phi[ci] * theta[language] * a_it
                rows.append({
                    "country": country,
                    "language": language,
                    "quarter": int(q),
                    "treated": int(treated[ci]),
                    "post": post,
                    "phi_i": phi[ci],
                    "theta_c": theta[language],
                    "x": x,
                    "base": alpha[ci] + beta_t[ti] + delta[li],
                    "treated_post": int(treated[ci] == 1 and post == 1),
                    "post_trend": max(q - 11, 0) / 5.0,
                })
    panel = pd.DataFrame(rows)

    d_country = pd.get_dummies(panel["country"], prefix="country", drop_first=True, dtype=float)
    d_quarter = pd.get_dummies(panel["quarter"], prefix="q", drop_first=True, dtype=float)
    d_language = pd.get_dummies(panel["language"], prefix="lang", drop_first=True, dtype=float)
    base_X = np.column_stack([
        np.ones(len(panel)),
        d_country.to_numpy(),
        d_quarter.to_numpy(),
        d_language.to_numpy(),
        panel["x"].to_numpy(),
    ])
    x_index = base_X.shape[1] - 1
    xtx_inv = np.linalg.inv(base_X.T @ base_X)
    projection = xtx_inv @ base_X.T
    return panel, base_X, projection, xtx_inv, x_index


def estimate_beta(y, X, projection, xtx_inv, x_index):
    beta_hat = projection @ y
    resid = y - X @ beta_hat
    n, k = X.shape
    sigma2 = float(resid @ resid / (n - k))
    se = float(np.sqrt(sigma2 * xtx_inv[x_index, x_index]))
    return float(beta_hat[x_index]), se


def run_monte_carlo(exposure_df, seed=SEED + 4, reps=500):
    panel, X, projection, xtx_inv, x_index = build_panel_design(exposure_df, seed=seed)
    rng = np.random.default_rng(seed + 100)

    x = panel["x"].to_numpy()
    base = panel["base"].to_numpy()
    treated_post = panel["treated_post"].to_numpy()
    post_trend = panel["post_trend"].to_numpy()
    theta = panel["theta_c"].to_numpy()
    phi = panel["phi_i"].to_numpy()
    post = panel["post"].to_numpy()

    scenarios = {
        "Exogenous treatment": np.zeros(len(panel)),
        "Treated post-trend": 0.22 * treated_post * post_trend,
        "Language-time shock": 0.15 * theta * post,
        "Country-language absorption": 0.18 * x * (phi - 1.0),
    }

    all_draws = []
    summary_rows = []
    true_beta = 1.0

    for scenario, confound in scenarios.items():
        estimates = []
        ses = []
        covered = []
        for _ in range(reps):
            eps = rng.normal(0.0, 0.75, size=len(panel))
            y = base + true_beta * x + confound + eps
            bhat, se = estimate_beta(y, X, projection, xtx_inv, x_index)
            estimates.append(bhat)
            ses.append(se)
            covered.append((bhat - 1.96 * se <= true_beta) and (true_beta <= bhat + 1.96 * se))
            all_draws.append({"scenario": scenario, "beta_hat": bhat})

        estimates = np.array(estimates)
        ses = np.array(ses)
        summary_rows.append({
            "Scenario": scenario,
            "True beta": true_beta,
            "Mean estimate": estimates.mean(),
            "Bias": estimates.mean() - true_beta,
            "RMSE": np.sqrt(np.mean((estimates - true_beta) ** 2)),
            "Mean SE": ses.mean(),
            "Coverage 95%": np.mean(covered),
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(all_draws)


def plot_monte_carlo(mc_draws):
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    scenarios = list(mc_draws["scenario"].drop_duplicates())
    data = [mc_draws.loc[mc_draws["scenario"] == s, "beta_hat"].to_numpy() for s in scenarios]
    parts = ax.violinplot(data, vert=False, showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#8ecae6")
        body.set_edgecolor("#12355b")
        body.set_alpha(0.85)
    for pos, vals in enumerate(data, start=1):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([q1, q3], [pos, pos], color="#12355b", lw=3)
        ax.scatter([med], [pos], color="#b00020", s=24, zorder=3)
    ax.set_yticks(np.arange(1, len(scenarios) + 1))
    ax.set_yticklabels(scenarios)
    ax.axvline(1.0, color="#b00020", lw=2, ls="--", label="True separable effect")
    ax.set_xlabel("Estimated coefficient on treatment exposure")
    ax.set_ylabel("")
    ax.set_title("Monte Carlo recovery of the fixed-effects estimator")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="lower right")
    savefig(os.path.join(FIG_DIR, "sim_monte_carlo_bias.png"))


def write_section_tex():
    section = r"""
\section{Simulation Evidence on the Theoretical Mechanism}
\label{sec:simulation_mechanism}

This section uses simulations as a bridge between the task-level model and the panel specification. The simulations are not used as causal evidence. Instead, they clarify the model's mechanics and show when the empirical specification recovers the structural effect embedded in the data-generating process.

\subsection{Task Exposure and Entry Thresholds}

The first simulation generates task-level human productivity $h_c(z)$ and ChatGPT productivity $a_c(z)$ for each programming language. For each language, we compute the automated-task set $\mathcal{Z}^{auto}_c=\{z:a_c(z)>h_c(z)\}$, the automated-task share $\tau_c$, the average relative saving $\rho_c$, and the exposure index $\theta_c=\tau_c\rho_c$. Figure~\ref{fig:sim_language_exposure} reports the simulated exposure measures by language.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.82\linewidth]{output/figures/simulations/sim_language_exposure.png}
\caption{Simulated task exposure by programming language}
\label{fig:sim_language_exposure}
\end{figure}

The second simulation maps the productivity gain into the entry margin. The model implies that access to AI lowers the entry threshold from $\eta_c^*(0)$ to $\eta_c^*(1)=\eta_c^*(0)-\tau_c\rho_c$. Figure~\ref{fig:sim_threshold_entry} illustrates the mass of marginal entrants located between the old and new thresholds.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.78\linewidth]{output/figures/simulations/sim_threshold_entry.png}
\caption{Lower entry threshold induced by AI productivity gains}
\label{fig:sim_threshold_entry}
\end{figure}

Combining language exposure with country-level absorption produces the prediction $\Delta Y_{ic}\approx \phi_i\theta_c$. Figure~\ref{fig:sim_theta_entry_relationship} shows that predicted entry is increasing in language exposure, with steeper responses in high-absorption countries.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.78\linewidth]{output/figures/simulations/sim_theta_entry_relationship.png}
\caption{Exposure and absorption jointly determine predicted entry}
\label{fig:sim_theta_entry_relationship}
\end{figure}

\subsection{Sensitivity to the Covariance Assumption}

The simplified model imposes $\operatorname{Cov}_{\mathcal{Z}^{auto}_c}(h_c,s_c)=0$, which yields $\Delta y_{jc}=\tau_c\rho_c$ after normalizing $\bar h_c=1$. The exact normalized gain is
\[
\Delta y_{jc}=\tau_c\left[\rho_c+\operatorname{Cov}_{\mathcal{Z}^{auto}_c}(h_c,s_c)\right].
\]
Figure~\ref{fig:sim_covariance_sensitivity} varies the correlation between human productivity and AI relative savings. The simplified exposure index closely tracks the exact gain near zero covariance, but it understates or overstates the gain when the covariance term becomes economically meaningful.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.78\linewidth]{output/figures/simulations/sim_covariance_sensitivity.png}
\caption{Sensitivity to the covariance assumption}
\label{fig:sim_covariance_sensitivity}
\end{figure}

\subsection{Monte Carlo Recovery of the Panel Estimator}

The final simulation generates panel data from
\[
Y_{ict}=\alpha_i+\beta_t+\delta_c+\theta_c\phi_i A_{it}+\eta_{ict},
\]
and estimates the same fixed-effects specification used in the empirical design. Table~\ref{tab:sim_monte_carlo} reports 500 Monte Carlo replications. Under exogenous treatment assignment, the estimator is centered around the true effect. When simulated shocks violate the identifying assumptions, the estimated coefficient becomes biased, which clarifies the role of the event-study, placebo, and robustness exercises in the empirical analysis.

\input{output/tables/simulations/simulation_monte_carlo.tex}
"""
    path = os.path.join(TAB_DIR, "simulation_section.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(section.strip() + "\n")


def main():
    ensure_dirs()
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })

    exposure_df = simulate_language_exposure()
    exposure_df.to_csv(os.path.join(TAB_DIR, "simulation_language_exposure.csv"), index=False)
    plot_language_exposure(exposure_df)

    exposure_table = exposure_df[[
        "language", "tau", "rho", "theta", "cov_norm", "exact_gain_norm"
    ]].rename(columns={
        "language": "Language",
        "tau": "$\\tau_c$",
        "rho": "$\\rho_c$",
        "theta": "$\\theta_c$",
        "cov_norm": "Cov.",
        "exact_gain_norm": "Exact gain",
    })
    dataframe_to_latex_table(
        exposure_table,
        caption=r"Simulated task exposure by language",
        label="tab:sim_language_exposure",
        path=os.path.join(TAB_DIR, "simulation_language_exposure.tex"),
    )

    plot_threshold_entry()

    entry_df = simulate_country_language_entry(exposure_df)
    entry_df.to_csv(os.path.join(TAB_DIR, "simulation_country_language_entry.csv"), index=False)
    plot_theta_entry_relationship(entry_df)

    cov_df = covariance_sensitivity()
    cov_df.to_csv(os.path.join(TAB_DIR, "simulation_covariance_sensitivity.csv"), index=False)
    plot_covariance_sensitivity(cov_df)

    cov_table = cov_df.loc[cov_df["corr_input"].isin([-0.8, -0.4, 0.0, 0.4, 0.8]), [
        "corr_input", "theta_simplified", "exact_gain_norm", "relative_gap_pct"
    ]].copy()
    cov_table = cov_table.rename(columns={
        "corr_input": "Corr.",
        "theta_simplified": "Simplified",
        "exact_gain_norm": "Exact",
        "relative_gap_pct": "Gap (%)",
    })
    dataframe_to_latex_table(
        cov_table,
        caption=r"Sensitivity of the gain index to the covariance assumption",
        label="tab:sim_covariance",
        path=os.path.join(TAB_DIR, "simulation_covariance_sensitivity.tex"),
    )

    mc_summary, mc_draws = run_monte_carlo(exposure_df, reps=500)
    mc_summary.to_csv(os.path.join(TAB_DIR, "simulation_monte_carlo.csv"), index=False)
    mc_draws.to_csv(os.path.join(TAB_DIR, "simulation_monte_carlo_draws.csv"), index=False)
    plot_monte_carlo(mc_draws)

    dataframe_to_latex_table(
        mc_summary,
        caption=r"Monte Carlo recovery of the fixed-effects estimator",
        label="tab:sim_monte_carlo",
        path=os.path.join(TAB_DIR, "simulation_monte_carlo.tex"),
    )

    write_section_tex()

    print("Simulation outputs written to:")
    print(f"  Figures: {FIG_DIR}")
    print(f"  Tables:  {TAB_DIR}")


if __name__ == "__main__":
    main()
