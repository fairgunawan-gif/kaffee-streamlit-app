import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests

BBC_URL = "https://www.bbc.com/sport/football/tables"

st.set_page_config(layout="wide")


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_table() -> pd.DataFrame:
    """Fetch the Premier League table from BBC Sport (no browser required)."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(BBC_URL, headers=headers, timeout=15)
    resp.raise_for_status()

    tables = pd.read_html(resp.text)
    if not tables:
        raise ValueError("No tables found on the page")
    df = tables[0]
    df.columns = [str(c).strip() for c in df.columns]

    # Normalise the team column name
    for col in df.columns:
        if col.lower() in ("club", "team"):
            df = df.rename(columns={col: "Team"})
            break

    # Drop unnamed index columns
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df


import re as _re

W, D, L = 3, 1, 0


def _parse_results(form_str):
    """
    Extract a list of result letters from BBC Sport form strings.
    BBC encodes each result as e.g. 'WResult Win', 'DResult Draw', 'LResult Loss'
    all concatenated together. We pull the letter immediately before 'Result'.
    Falls back to reading plain W/D/L characters for any other format.
    """
    s = str(form_str)
    if "Result" in s:
        return _re.findall(r"([WDL])Result", s)
    return [c for c in s.upper() if c in "WDL"]


def form_points(form_str):
    """Sum points for last 6 games (W=3, D=1, L=0)."""
    results = _parse_results(form_str)[:6]   # safety: only last 6
    return sum(W if r == "W" else D if r == "D" else L for r in results)


def form_label(form_str):
    """Return a clean readable label like 'W D L W D W'."""
    return " ".join(_parse_results(form_str)[:6])


# ── UI ───────────────────────────────────────────────────────────────────────

st.title("⚽ Premier League Table — Live from BBC Sport")
st.caption(f"Source: {BBC_URL}")

with st.spinner("Fetching live table…"):
    try:
        df = fetch_table()
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()

full_df = df.copy()
points_col = next((c for c in full_df.columns if c.lower() in ("pts", "points")), None)

if points_col and "Team" in full_df.columns:
    cols = list(full_df.columns)
    cols.remove(points_col)
    team_index = cols.index("Team")
    cols.insert(team_index + 1, points_col)
    full_df = full_df[cols]

# Drop the last column from the displayed full table
full_df_display = full_df.iloc[:, :-1]

# Row 1: Full Table in middle column (1 : 4 : 1)
col_ft_l, col_ft_m, col_ft_r = st.columns([1, 4, 1])
with col_ft_m:
    st.subheader("Full Table")
    st.dataframe(full_df_display, use_container_width=True, hide_index=True)

# ── Team filter: multiselect in left column, Selected Team Stats in middle (same row) ──

if "Team" in df.columns:
    all_teams = df["Team"].dropna().unique()

    col_sel_l, col_sel_m, col_sel_r = st.columns([1, 4, 1])
    with col_sel_l:
        selected_teams = st.multiselect(
            "Select between 1 and 4 teams",
            options=all_teams,
            default=list(all_teams[:1]),
            max_selections=4,
        )

    if selected_teams:
        team_rows = df[df["Team"].isin(selected_teams)].copy()

        points_col = next(
            (c for c in team_rows.columns if c.lower() in ("pts", "points")),
            None,
        )

        form_col = team_rows.columns[-1]
        if "form" not in form_col.lower():
            form_col = next(
                (c for c in team_rows.columns if "form" in c.lower()),
                None,
            )

        last6_col_name = "Last 6 Pts"
        if form_col is not None:
            team_rows[last6_col_name] = team_rows[form_col].apply(form_points)
        else:
            team_rows[last6_col_name] = None

        cols = list(team_rows.columns)
        ordered_cols = []

        if "Team" in cols:
            ordered_cols.append("Team")

        if points_col and points_col in cols and points_col not in ordered_cols:
            ordered_cols.append(points_col)

        if last6_col_name not in ordered_cols:
            ordered_cols.append(last6_col_name)

        remaining_cols = [
            c for c in cols if c not in ordered_cols and c != form_col
        ]
        ordered_cols.extend(remaining_cols)

        selected_stats_df = team_rows[ordered_cols]

    with col_sel_m:
        st.subheader("Selected Team Stats")
        if selected_teams:
            st.dataframe(selected_stats_df, use_container_width=True, hide_index=True)
        else:
            st.info("Select at least one team to see stats.")
else:
    st.warning("Could not identify a 'Team' column in the scraped data.")
    st.write("Raw columns found:", list(df.columns))

# ── Form ranking bar chart ────────────────────────────────────────────────────

# Use the last column of the full table as the form column;
# fall back to any column whose name contains "form"
FORM_COL = df.columns[-1]
if "form" not in FORM_COL.lower():
    FORM_COL = next((c for c in df.columns if "form" in c.lower()), None)

if FORM_COL and "Team" in df.columns:
    df["Form_pts"]   = df[FORM_COL].apply(form_points)
    df["Form_label"] = df[FORM_COL].apply(form_label)

    form_ranking = (
        df[["Team", "Form_label", "Form_pts"]]
        .rename(columns={"Form_label": "Form"})
        .sort_values("Form_pts", ascending=False)
        .reset_index(drop=True)
    )
    form_ranking.insert(0, "Form_Rank", form_ranking.index + 1)

    def bar_colour(pts):
        if pts >= 12:
            return "#2ecc71"   # green
        elif pts >= 6:
            return "#f39c12"   # amber
        else:
            return "#e74c3c"   # red

    colours = [bar_colour(p) for p in form_ranking["Form_pts"]]

    # Match Streamlit table font: sans-serif, table-like size (~14px / 10pt)
    table_font = "DejaVu Sans"  # concrete font; "sans-serif" breaks legend prop
    table_fontsize = 10

    fig, ax = plt.subplots(figsize=(9, len(form_ranking) * 0.42 + 1))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plt.rcParams["font.family"] = table_font
    plt.rcParams["font.size"] = table_fontsize

    bars = ax.barh(
        form_ranking["Team"][::-1],
        form_ranking["Form_pts"][::-1],
        color=colours[::-1],
        edgecolor="white",
        height=0.7,
    )

    for bar, label, pts in zip(
        bars,
        form_ranking["Form"][::-1],
        form_ranking["Form_pts"][::-1],
    ):
        ax.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{label}  ({pts} pts)",
            va="center", ha="left", fontsize=table_fontsize, fontfamily=table_font, color="#333333",
        )

    ax.set_xlim(0, 22)
    ax.set_xlabel("Form Points", fontsize=table_fontsize, fontfamily=table_font)
    ax.set_ylabel(None)
    ax.set_title("Last-6-Game Form Ranking", fontsize=table_fontsize, fontweight="bold", fontfamily=table_font)
    ax.tick_params(axis="both", labelsize=table_fontsize)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(plt.MultipleLocator(3))
    ax.axvline(x=9, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)

    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="Strong (≥ 12 pts)"),
        mpatches.Patch(color="#f39c12", label="Mixed (6–11 pts)"),
        mpatches.Patch(color="#e74c3c", label="Poor (< 6 pts)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=table_fontsize, framealpha=0.5)

    plt.tight_layout()

    col_fr_l, col_fr_m, col_fr_r = st.columns([1, 4, 1])
    with col_fr_m:
        st.subheader("📊 Form Ranking — Last 6 Games")
        st.caption("Points: W = 3 · D = 1 · L = 0  (max 18)")
        st.pyplot(fig)

    # ── Season points over last 6 games (line chart) ───────────────────────────

    points_col = next(
        (c for c in df.columns if c.lower() in ("pts", "points")),
        None,
    )

    def last6_season_points(form_str, current_pts):
        """Return cumulative season points for the last 6 games (oldest → newest)."""
        try:
            total_now = int(current_pts)
        except (TypeError, ValueError):
            return []

        results = _parse_results(form_str)[:6]
        if not results:
            return []

        per_game = [
            W if r == "W" else D if r == "D" else L
            for r in results
        ]
        start = total_now - sum(per_game)
        vals = []
        running = start
        for p in per_game:
            running += p
            vals.append(running)
        return vals

    # build rows for line chart, optionally limiting to selected teams
    line_rows = []
    # determine which teams to include based on the selector
    filter_teams = None
    if "selected_teams" in locals() and selected_teams:
        filter_teams = set(selected_teams)

    if points_col:
        for _, row in df.iterrows():
            team = row.get("Team")
            if not team:
                continue
            # skip teams not selected (when the filter is active)
            if filter_teams is not None and team not in filter_teams:
                continue
            form_str = row.get(FORM_COL)
            current_pts = row.get(points_col)
            seq = last6_season_points(form_str, current_pts)
            if not seq:
                continue
            last_val = seq[-1]
            for idx, val in enumerate(seq, start=1):
                line_rows.append(
                    {
                        "Team": team,
                        "Game": idx,
                        "Points": val,
                        "LastPoints": last_val,
                    }
                )

    if line_rows:
        line_df = pd.DataFrame(line_rows)
        team_order = (
            line_df.groupby("Team")["LastPoints"]
            .max()
            .sort_values(ascending=False)
            .index.tolist()
        )

        fig_line, ax_line = plt.subplots(figsize=(9, 4.5))

        for team in team_order:
            sub = line_df[line_df["Team"] == team]
            ax_line.plot(
                sub["Game"],
                sub["Points"],
                marker="o",
                linewidth=1.2,
                label=team,
            )

        ax_line.set_xticks(range(1, 7))
        ax_line.set_xlabel("Last 6 games (oldest → newest)")
        ax_line.set_ylabel("Season points")
        ax_line.set_title("Season Points over Last 6 Games")
        ax_line.grid(axis="y", alpha=0.2)
        ax_line.spines[["top", "right"]].set_visible(False)
        ax_line.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
        plt.tight_layout()

        col_ln_l, col_ln_m, col_ln_r = st.columns([1, 4, 1])
        with col_ln_m:
            st.subheader("Season Points — Last 6 Games")
            st.pyplot(fig_line)
    else:
        # no rows to plot, possibly because the user cleared the selection
        col_ln_l, col_ln_m, col_ln_r = st.columns([1, 4, 1])
        with col_ln_m:
            if "selected_teams" in locals() and selected_teams:
                st.info("No data available for the selected teams' last six games.")
            else:
                st.info("No season points data available to chart.")
else:
    col_fr_l, col_fr_m, col_fr_r = st.columns([1, 4, 1])
    with col_fr_m:
        st.info("No Form column found in the scraped data — chart unavailable.")

# ── Balloons ──────────────────────────────────────────────────────────────────

if st.button("🎈 Send balloons!"):
    st.balloons()


# Whenever you change the code:
# In VS Code:
# git add .
# git commit -m "Describe your change"
# git push