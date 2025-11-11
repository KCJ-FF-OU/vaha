# vaha_app.py
# -----------------------------------------
# Interaktivní dilematická hra „VáHa“ (verze 1.6)
# – normalizované R (R_norm ∈ [0,1])
# – kompatibilní se strukturou CSV: typ;rok_XXXX_text;aut_XXXX;emp_XXXX
# -----------------------------------------

import streamlit as st
st.set_page_config(page_title="VáHa", layout="wide")
st.markdown("<title>VáHa – autenticita versus empatie</title>", unsafe_allow_html=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------- NAČTENÍ DAT ----------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    # Oprava typografických znamének minus
    df = df.replace("−", "-", regex=True)
    df = df.replace("–", "-", regex=True)
    df = df.replace("—", "-", regex=True)

    # Převod aut_ a emp_ sloupců na číselné hodnoty
    for col in df.columns:
        if col.startswith("aut_") or col.startswith("emp_"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df

DATA_PATH = "vaha_full.csv"
df = load_data(DATA_PATH)

# ---------- ROČNÍKY ----------
YEAR_COLS = [
    {"year": 1968, "text": "rok_1968_text", "aut": "aut_1968", "emp": "emp_1968"},
    {"year": 1970, "text": "rok_1970_text", "aut": "aut_1970", "emp": "emp_1970"},
    {"year": 1977, "text": "rok_1977_text", "aut": "aut_1977", "emp": "emp_1977"},
    {"year": 1989, "text": "rok_1989_text", "aut": "aut_1989", "emp": "emp_1989"},
]

# ---------- SESSION STATE ----------
if "prefix" not in st.session_state:
    st.session_state.prefix = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "finished" not in st.session_state:
    st.session_state.finished = False

def reset_game():
    st.session_state.prefix = ""
    st.session_state.history = []
    st.session_state.finished = False

# ---------- FUNKCE ----------
def candidates_for_step(df, prefix, step_idx):
    year_conf = YEAR_COLS[step_idx]
    txt_col, aut_col, emp_col = year_conf["text"], year_conf["aut"], year_conf["emp"]
    level = len(prefix) + 1
    sub = df[df["typ"].str.startswith(prefix)]
    sub = sub[sub["typ"].str.len() >= level]

    options = {}
    for _, row in sub.iterrows():
        next_letter = row["typ"][len(prefix)]
        if next_letter not in options:
            options[next_letter] = (row[txt_col], int(row[aut_col]), int(row[emp_col]))
    return options

def pick_final_row(df, full_typ):
    m = df[df["typ"] == full_typ]
    if len(m) >= 1:
        return m.iloc[0]
    return None

def step_ui(step_idx):
    year = YEAR_COLS[step_idx]["year"]
    st.subheader(f"{step_idx+1}/4 • Rok {year}")

    options = candidates_for_step(df, st.session_state.prefix, step_idx)
    if not options:
        st.warning("Pro tento krok nejsou dostupné žádné větve. Zkus hru restartovat.")
        return

    for letter in sorted(options.keys()):
        text, aut, emp = options[letter]
        with st.container(border=True):
            st.markdown(f"**Varianta {letter}**")
            st.write(text)
            if st.button(f"Zvolit {letter}", key=f"btn_{step_idx}_{letter}"):
                st.session_state.history.append({
                    "year": year,
                    "letter": letter,
                    "text": text,
                    "aut": int(aut),
                    "emp": int(emp),
                })
                st.session_state.prefix += letter
                if len(st.session_state.prefix) == len(YEAR_COLS):
                    st.session_state.finished = True
                st.rerun()

# ---------- ROZVRŽENÍ ----------
left, right = st.columns([2, 1])

with left:
    st.title("VáHa – autenticita versus empatie")
    st.write(
        "Čekají tě čtyři roky, čtyři milníky, čtyři rozhodnutí. "
        "Každé tvé rozhodnutí se hodnotí co do autenticity (AUT) a empatie (EMP). "
        "Jako autentická se hodnotí ta rozhodnutí, která vycházejí z tvého přesvědčení. "
        "Za empatická považujeme rozhodnutí, která mají za cíl ochránit tvé blízké. " 
        "Na konci uvidíš shrnutí své cesty a sílu tvého postoje. "
        "Maximální hodnota síly postoje je 1, minimální 0. "
        "Jde především o to, jak konzistentní byly tvoje volby. "
        "Nejedná se o morální soud. "
        "Poznámka: VáHa je hra. Nejde o učební platformu dějepisu. Pracuje s alternativní historií, ale snaží se o věrohodnost a smysluplnost."
    )
    st.divider()

    if not st.session_state.finished:
        step_ui(len(st.session_state.prefix))
    else:
        full_typ = st.session_state.prefix
        row = pick_final_row(df, full_typ)
        st.success(f"Hotovo! Tvoje cesta: **{full_typ}**")

        with st.container(border=True):
            st.markdown("### Tvoje volby")
            for rec in st.session_state.history:
                st.markdown(f"- **{rec['year']} – {rec['letter']}**: {rec['text']}")

            total_aut = sum(r["aut"] for r in st.session_state.history)
            total_emp = sum(r["emp"] for r in st.session_state.history)

            # --- Síla postoje: R (raw) a normalizované R_norm ∈ [0,1] ---
            n_steps = len(YEAR_COLS)
            R = np.sqrt(total_aut**2 + total_emp**2)
            Rmax = n_steps * np.sqrt(2)
            R_norm = (R / Rmax) if Rmax > 0 else 0.0

            # --- Prahy interpretace ve škále R_norm ---
            RN_T1 = 2 / Rmax
            RN_T2 = 3 / Rmax

            # --- Interpretace výsledků ---
            def interpretace_vyroku(A, E, Rn):
                if A > E:
                    if Rn < RN_T1:
                        return "Neřídím se davem – žádným."
                    elif Rn < RN_T2:
                        return "Zvolil jsem si autenticitu, ale nezaslepenou."
                    else:
                        return "Nedělám morální kompromisy. Ani za cenu následků."
                elif E > A:
                    if Rn < RN_T1:
                        return "Vyhrává u mě rodina, ale mám i své přesvědčení."
                    elif Rn < RN_T2:
                        return "Nemůžu se na rodinu vykašlat. Ale nepřestávám ani bojovat."
                    else:
                        return "Moje rodina = moje priorita. Systém se může zbořit sám."
                else:
                    return "Zvažuju, balancuju, snažím se volit podle situace."

            def urci_smer(A, E):
                if A > E:
                    return "Autentický", "🟥"
                elif E > A:
                    return "Empatický", "🟦"
                else:
                    return "Vyrovnaný", "⚪"

            smer, barva = urci_smer(total_aut, total_emp)
            veta = interpretace_vyroku(total_aut, total_emp, R_norm)

            st.markdown("### Souhrnné skóre")
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.metric("AUT celkem", f"{total_aut:+d}")
            with c2:
                st.metric("EMP celkem", f"{total_emp:+d}")
            with c3:
                st.metric("Síla postoje", f"{R_norm:.2f}")

            st.progress(min(max(R_norm, 0.0), 1.0))
            st.markdown(f"**Směr:** {barva} {smer}")
            st.markdown(f"### {barva} _{veta}_")

        # === Prémie: graf vývoje rozhodnutí v čase ===
        st.markdown("### 📈 Vývoj tvých rozhodnutí v čase")

        traj_df = pd.DataFrame(st.session_state.history)
        if not traj_df.empty:
            traj_df = traj_df.sort_values("year")
            traj_df["A_kumul"] = traj_df["aut"].cumsum()
            traj_df["E_kumul"] = traj_df["emp"].cumsum()

            fig, ax = plt.subplots()
            ax.plot(traj_df["year"], traj_df["A_kumul"], marker="o", linewidth=2, label="Autenticita (A)")
            ax.plot(traj_df["year"], traj_df["E_kumul"], marker="o", linewidth=2, label="Empatie (E)")
            ax.set_xlabel("Rok")
            ax.set_ylabel("Kumulativní skóre")
            ax.legend(frameon=False)
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig)
        else:
            st.caption("Zatím nejsou k dispozici žádná data pro zobrazení trajektorie.")

        st.button("Hrát znovu", on_click=reset_game, type="primary")

with right:
    st.markdown("#### Stav")
    st.write(f"**Kód cesty:** `{st.session_state.prefix or '—'}`")
    st.progress(len(st.session_state.prefix) / len(YEAR_COLS))
    st.button("Restartovat", on_click=reset_game)

