"""
Simulateur — Salaire requis (France, salarié)
Design:
- Fond photo (assets/chateau_meung.jpg) + overlay sombre
- Bandeau premium (hero) avec le titre demandé + citation
- UI épurée (4 onglets), pas d'export/import

Immo:
- Si type=immo et prix_bien>0 : capital financé = prix_bien - apport (+ notaire optionnel)
- Sinon : capital financé = capital ; si apport>0, on le déduit du capital (warning)

Simplification dépenses:
- Pas de notes, pas de dates.
- Dépense "ponctuel" = étalée sur 12 mois (warning).
"""

from __future__ import annotations

import base64
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------
# Config
# -----------------------
APP_TITLE = "Simulateur de salaire — Train de vie (France)"
CURRENCY = "€"
BACKGROUND_IMAGE_PATH = "assets/chateau_meung.jpg"

FREQ_OPTIONS = ["mensuel", "hebdo", "trimestriel", "annuel", "ponctuel"]
LEVEL_OPTIONS = ["Essentiel", "Confort"]
LOAN_TYPE_OPTIONS = ["immo", "auto", "conso", "etudiant", "autre"]
INS_TYPE_OPTIONS = ["Aucune", "€/mois", "%/an sur capital"]


# -----------------------
# Design helpers
# -----------------------
def _img_to_data_uri(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def inject_css():
    bg = _img_to_data_uri(BACKGROUND_IMAGE_PATH)

    if bg:
        background_css = f"""
        .stApp {{
          background-image:
            linear-gradient(rgba(8,10,14,0.82), rgba(8,10,14,0.82)),
            url("{bg}");
          background-size: cover;
          background-attachment: fixed;
          background-position: center;
        }}
        """
    else:
        background_css = """
        .stApp {
          background: radial-gradient(1200px 800px at 20% 10%, #1f2937 0%, #0b0f14 55%, #070a0f 100%);
        }
        """

    st.markdown(
        f"""
        <style>
        {background_css}

        #MainMenu {{visibility: hidden;}}
        header {{visibility: hidden;}}
        footer {{visibility: hidden;}}

        .block-container {{
          padding-top: 1.2rem;
          max-width: 1180px;
        }}

        .hero {{
          padding: 26px 28px;
          border-radius: 18px;
          background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.05));
          border: 1px solid rgba(255,255,255,0.14);
          backdrop-filter: blur(10px);
          box-shadow: 0 18px 50px rgba(0,0,0,0.40);
          margin-bottom: 16px;
        }}
        .hero h1 {{
          color: #fff;
          margin: 0;
          font-size: 44px;
          line-height: 1.05;
          font-weight: 900;
          letter-spacing: -0.6px;
        }}
        .hero p {{
          color: rgba(255,255,255,0.70);
          margin: 10px 0 0;
          font-size: 14px;
        }}

        .card {{
          padding: 16px 18px;
          border-radius: 16px;
          background: rgba(255,255,255,0.06);
          border: 1px solid rgba(255,255,255,0.10);
          backdrop-filter: blur(10px);
          box-shadow: 0 10px 30px rgba(0,0,0,0.30);
        }}

        .stTabs [data-baseweb="tab-list"] {{
          gap: 8px;
          background: rgba(255,255,255,0.04);
          padding: 8px;
          border-radius: 16px;
          border: 1px solid rgba(255,255,255,0.10);
        }}
        .stTabs [data-baseweb="tab"] {{
          border-radius: 12px;
          padding: 10px 14px;
          color: rgba(255,255,255,0.75);
          background: transparent;
        }}
        .stTabs [aria-selected="true"] {{
          background: rgba(255,255,255,0.12) !important;
          color: #fff !important;
          border: 1px solid rgba(255,255,255,0.16);
        }}

        h2, h3, h4, p, label, .stMarkdown {{
          color: rgba(255,255,255,0.92) !important;
        }}
        .stCaption {{
          color: rgba(255,255,255,0.70) !important;
        }}

        [data-testid="stDataFrame"], [data-testid="stTable"] {{
          background: rgba(255,255,255,0.04);
          border-radius: 14px;
          border: 1px solid rgba(255,255,255,0.10);
          overflow: hidden;
        }}

        [data-testid="stAlert"] {{
          border-radius: 14px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero():
    st.markdown(
        """
        <div class="hero">
          <h1>Simulateur de salaire, la richesse sinon rien.</h1>
          <p>« Ce qui ne se mesure pas ne s’améliore pas. » — Peter Drucker</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card_start():
    st.markdown('<div class="card">', unsafe_allow_html=True)


def card_end():
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------
# Utils
# -----------------------
def fmt_eur(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:,.0f} {CURRENCY}".replace(",", " ")


# -----------------------
# Finance (prêt)
# -----------------------
def pmt(rate: float, nper: int, pv: float) -> float:
    if nper <= 0:
        return 0.0
    if abs(rate) < 1e-12:
        return pv / nper
    return pv * (rate / (1 - (1 + rate) ** (-nper)))


def loan_monthly_payment(capital: float, annual_rate_pct: float, years: float) -> float:
    n = int(round(years * 12))
    r = (annual_rate_pct / 100.0) / 12.0
    return float(pmt(r, n, capital))


def amortization_schedule(capital: float, annual_rate_pct: float, years: float) -> pd.DataFrame:
    n = int(round(years * 12))
    r = (annual_rate_pct / 100.0) / 12.0
    payment = loan_monthly_payment(capital, annual_rate_pct, years)

    balance = capital
    rows = []
    for k in range(1, n + 1):
        interest = balance * r if r > 0 else 0.0
        principal = payment - interest
        if principal > balance:
            principal = balance
            payment_eff = principal + interest
        else:
            payment_eff = payment
        balance = max(0.0, balance - principal)
        rows.append(
            {"Mois": k, "Mensualité": payment_eff, "Intérêts": interest, "Principal": principal, "Capital restant dû": balance}
        )
        if balance <= 1e-6:
            break
    return pd.DataFrame(rows)


# -----------------------
# Schemas (simplifiés)
# -----------------------
EXPENSE_COLS = ["nom", "categorie", "montant", "frequence", "niveau", "actif"]

LOAN_COLS = [
    "nom", "type",
    "capital",
    "prix_bien",
    "apport",
    "inclure_frais_notaire",
    "frais_notaire_pct",
    "taux_annuel_pct", "duree_annees",
    "assurance_mode", "assurance_mensuelle", "assurance_taux_annuel_pct",
    "frais", "etaler_frais"
]


def ensure_expense_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame(columns=EXPENSE_COLS)
    for c in EXPENSE_COLS:
        if c not in df.columns:
            df[c] = None

    df["montant"] = pd.to_numeric(df["montant"], errors="coerce").fillna(0.0)
    df["frequence"] = df["frequence"].fillna("mensuel")
    df["niveau"] = df["niveau"].fillna("Essentiel")
    df["actif"] = df["actif"].fillna(True).astype(bool)
    df["categorie"] = df["categorie"].fillna("Autres")
    df["nom"] = df["nom"].fillna("")
    return df[EXPENSE_COLS].copy()


def ensure_loan_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        df = pd.DataFrame(columns=LOAN_COLS)
    for c in LOAN_COLS:
        if c not in df.columns:
            df[c] = None

    num_cols = [
        "capital", "prix_bien", "apport", "frais_notaire_pct",
        "taux_annuel_pct", "duree_annees",
        "assurance_mensuelle", "assurance_taux_annuel_pct", "frais"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["type"] = df["type"].fillna("autre")
    df["assurance_mode"] = df["assurance_mode"].fillna("Aucune")
    df["etaler_frais"] = df["etaler_frais"].fillna(True).astype(bool)
    df["inclure_frais_notaire"] = df["inclure_frais_notaire"].fillna(False).astype(bool)
    df["nom"] = df["nom"].fillna("")
    return df[LOAN_COLS].copy()


# -----------------------
# Expenses mensualisation (sans dates)
# -----------------------
def monthlyize_expenses(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    df = ensure_expense_schema(df).copy()

    monthly_vals = []
    annual_vals = []

    for _, row in df.iterrows():
        amount = float(row["montant"] or 0.0)
        freq = str(row["frequence"]).strip().lower()

        if amount < 0:
            warnings.append(f"Montant négatif sur '{row['nom']}' (vérifie).")

        if freq not in FREQ_OPTIONS:
            warnings.append(f"Fréquence inconnue '{freq}' → traitée comme 'mensuel'.")
            freq = "mensuel"

        if freq == "mensuel":
            m = amount
        elif freq == "hebdo":
            m = amount * (52 / 12)
        elif freq == "trimestriel":
            m = amount / 3
        elif freq == "annuel":
            m = amount / 12
        elif freq == "ponctuel":
            # Sans dates : on mensualise par convention sur 12 mois.
            m = amount / 12
            warnings.append(f"Ponctuel '{row['nom']}' : étalé sur 12 mois (pas de dates).")
        else:
            m = amount

        monthly_vals.append(m)
        annual_vals.append(m * 12)

    df["mensuel_equiv"] = monthly_vals
    df["annuel_equiv"] = annual_vals
    return df, warnings


# -----------------------
# Loans compute (immo + apport)
# -----------------------
def compute_loans(df: pd.DataFrame) -> Tuple[pd.DataFrame, float, List[str]]:
    warnings: List[str] = []
    df = ensure_loan_schema(df).copy()

    if df.empty:
        for col in ["capital_finance_calc", "mensualite_hors_assurance", "assurance_calc",
                    "frais_mensuels_calc", "mensualite_totale"]:
            df[col] = []
        return df, 0.0, warnings

    capital_finance = []
    pays = []
    ins = []
    fees_m = []
    totals = []

    for _, r in df.iterrows():
        name = r["nom"] or "(sans nom)"
        loan_type = str(r["type"] or "autre").strip().lower()

        cap = float(r["capital"])
        price = float(r["prix_bien"])
        apport = float(r["apport"])
        incl_notary = bool(r["inclure_frais_notaire"])
        notary_pct = float(r["frais_notaire_pct"])

        cap_used = cap

        if loan_type == "immo":
            if price > 0:
                cap_used = max(0.0, price - apport)
                if apport > price:
                    warnings.append(f"Prêt immo '{name}': apport > prix du bien → capital financé mis à 0.")
                if incl_notary:
                    if notary_pct > 0:
                        cap_used += price * (notary_pct / 100.0)
                    else:
                        warnings.append(f"Prêt immo '{name}': frais de notaire cochés mais % = 0.")
            else:
                cap_used = cap
                if cap > 0 and apport > 0:
                    cap_used = max(0.0, cap - apport)
                    warnings.append(f"Prêt immo '{name}': prix du bien non renseigné → apport déduit du capital saisi.")
                if incl_notary and notary_pct > 0:
                    warnings.append(f"Prêt immo '{name}': frais de notaire non appliqués (prix du bien manquant).")

            if (cap > 0 or price > 0) and cap_used <= 0:
                warnings.append(f"Prêt immo '{name}': capital financé calculé à 0 (vérifie prix/apport/capital).")

        capital_finance.append(cap_used)

        rate = float(r["taux_annuel_pct"])
        years = float(r["duree_annees"])
        n = int(round(years * 12)) if years > 0 else 0

        if years <= 0 and cap_used > 0:
            warnings.append(f"Prêt '{name}': durée invalide (>0 requis).")

        pay = loan_monthly_payment(cap_used, rate, years) if (cap_used > 0 and years > 0) else 0.0

        mode = str(r["assurance_mode"] or "Aucune")
        if mode == "€/mois":
            insurance = float(r["assurance_mensuelle"])
        elif mode == "%/an sur capital":
            insurance = cap_used * (float(r["assurance_taux_annuel_pct"]) / 100.0) / 12.0
        else:
            insurance = 0.0

        fees = float(r["frais"])
        spread = bool(r["etaler_frais"])
        f_m = (fees / n) if (spread and n > 0) else 0.0

        total = pay + insurance + f_m

        pays.append(pay)
        ins.append(insurance)
        fees_m.append(f_m)
        totals.append(total)

    df["capital_finance_calc"] = capital_finance
    df["mensualite_hors_assurance"] = pays
    df["assurance_calc"] = ins
    df["frais_mensuels_calc"] = fees_m
    df["mensualite_totale"] = totals

    total_monthly = float(np.nansum(df["mensualite_totale"].values))
    return df, total_monthly, warnings


# -----------------------
# Salary logic
# -----------------------
def compute_need_net_after_ir(
    depenses_m: float,
    prets_m: float,
    epargne_fixe: float,
    epargne_pct_revenu: float,
    marge_imprevus_pct: float,
) -> Tuple[float, Dict[str, float]]:
    marge = marge_imprevus_pct * depenses_m
    base = depenses_m + prets_m + epargne_fixe + marge

    if epargne_pct_revenu >= 0.95:
        return float("nan"), {}

    need = base / (1 - epargne_pct_revenu) if epargne_pct_revenu > 0 else base

    breakdown = {
        "Dépenses": depenses_m,
        "Crédits": prets_m,
        "Épargne fixe": epargne_fixe,
        "Marge imprévus": marge,
        "Épargne (% revenu)": epargne_pct_revenu * need if epargne_pct_revenu > 0 else 0.0,
    }
    return need, breakdown


def net_before_ir_from_pas(net_after_ir: float, pas_rate: float) -> float:
    return net_after_ir / (1 - pas_rate) if pas_rate < 1 else float("nan")


def brut_from_ratio(net_before_ir: float, ratio_net_brut: float) -> float:
    return net_before_ir / ratio_net_brut if ratio_net_brut > 0 else float("nan")


def build_scenarios(exp_m: pd.DataFrame, loans_total_monthly: float, assumptions: Dict) -> Tuple[List[Dict], pd.DataFrame]:
    active = exp_m[exp_m["actif"] == True].copy()
    dep_total_m = float(active["mensuel_equiv"].sum())
    dep_ess_m = float(active[active["niveau"] == "Essentiel"]["mensuel_equiv"].sum())

    base_epargne = float(assumptions.get("epargne_fixe", 300.0))
    base_marge = float(assumptions.get("marge_imprevus_pct", 0.10))
    amb_bonus_epargne = float(assumptions.get("amb_bonus_epargne", 200.0))
    amb_bonus_marge = float(assumptions.get("amb_bonus_marge", 0.0))

    def _scenario(name: str, depenses_m: float, epargne_fix: float, marge_pct: float) -> Dict:
        need_after, breakdown = compute_need_net_after_ir(
            depenses_m=depenses_m,
            prets_m=loans_total_monthly,
            epargne_fixe=epargne_fix,
            epargne_pct_revenu=float(assumptions.get("epargne_pct_revenu", 0.0)),
            marge_imprevus_pct=marge_pct,
        )
        net_before = net_before_ir_from_pas(need_after, float(assumptions["pas_rate"]))
        brut = brut_from_ratio(net_before, float(assumptions["ratio_net_brut"]))
        return {
            "Scénario": name,
            "Net après IR (mensuel)": need_after,
            "Net avant IR (mensuel)": net_before,
            "Brut (mensuel)": brut,
            "Net après IR (annuel)": need_after * 12,
            "Net avant IR (annuel)": net_before * 12,
            "Brut (annuel)": brut * 12,
            "_breakdown": breakdown,
        }

    scenarios = [
        _scenario("Minimum", dep_ess_m, base_epargne, base_marge),
        _scenario("Confort", dep_total_m, base_epargne, base_marge),
        _scenario("Ambitieux", dep_total_m, base_epargne + amb_bonus_epargne, base_marge + amb_bonus_marge),
    ]
    scen_df = pd.DataFrame([{k: v for k, v in s.items() if not k.startswith("_")} for s in scenarios])
    return scenarios, scen_df


# -----------------------
# Defaults
# -----------------------
def default_expenses() -> pd.DataFrame:
    data = [
        {"nom": "Loyer", "categorie": "Logement", "montant": 1150.0, "frequence": "mensuel", "niveau": "Essentiel", "actif": True},
        {"nom": "Charges", "categorie": "Logement", "montant": 50.0, "frequence": "mensuel", "niveau": "Essentiel", "actif": True},
        {"nom": "Énergie (élec/gaz)", "categorie": "Logement", "montant": 90.0, "frequence": "mensuel", "niveau": "Essentiel", "actif": True},
        {"nom": "Internet + abonnements", "categorie": "Abonnements", "montant": 60.0, "frequence": "mensuel", "niveau": "Essentiel", "actif": True},
        {"nom": "Assurance habitation", "categorie": "Assurances", "montant": 18.0, "frequence": "mensuel", "niveau": "Essentiel", "actif": True},
        {"nom": "Courses", "categorie": "Alimentation", "montant": 450.0, "frequence": "mensuel", "niveau": "Essentiel", "actif": True},
        {"nom": "Restaurants", "categorie": "Alimentation", "montant": 100.0, "frequence": "mensuel", "niveau": "Confort", "actif": True},
        {"nom": "Transport (Navigo)", "categorie": "Transport", "montant": 90.0, "frequence": "mensuel", "niveau": "Essentiel", "actif": True},
        {"nom": "Assurance auto", "categorie": "Assurances", "montant": 80.0, "frequence": "mensuel", "niveau": "Confort", "actif": True},
        {"nom": "Vacances (mensualisées)", "categorie": "Loisirs", "montant": 1800.0, "frequence": "annuel", "niveau": "Confort", "actif": True},
    ]
    return ensure_expense_schema(pd.DataFrame(data))


def default_loans() -> pd.DataFrame:
    data = [
        {
            "nom": "Prêt immo (exemple)",
            "type": "immo",
            "capital": 0.0,
            "prix_bien": 170000.0,
            "apport": 20000.0,
            "inclure_frais_notaire": False,
            "frais_notaire_pct": 8.5,
            "taux_annuel_pct": 3.6,
            "duree_annees": 25.0,
            "assurance_mode": "Aucune",
            "assurance_mensuelle": 0.0,
            "assurance_taux_annuel_pct": 0.0,
            "frais": 0.0,
            "etaler_frais": True,
        }
    ]
    return ensure_loan_schema(pd.DataFrame(data))


def default_assumptions() -> Dict:
    return {
        "pas_rate": 0.081,          # 8,1%
        "ratio_net_brut": 0.78,
        "epargne_fixe": 300.0,
        "epargne_pct_revenu": 0.0,  # avancé
        "marge_imprevus_pct": 0.10,
        "amb_bonus_epargne": 200.0,
        "amb_bonus_marge": 0.00,
    }


# -----------------------
# State
# -----------------------
def init_state():
    if "expenses_df" not in st.session_state:
        st.session_state.expenses_df = default_expenses()
    if "loans_df" not in st.session_state:
        st.session_state.loans_df = default_loans()
    if "assumptions" not in st.session_state:
        st.session_state.assumptions = default_assumptions()


# -----------------------
# App
# -----------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_css()
    init_state()

    hero()

    st.caption(
        "On part du **net après IR (cash sur le compte)**, puis on remonte vers **net avant IR** et **brut**. "
        "PAS et ratio net→brut sont des approximations ajustables."
    )

    tab1, tab2, tab3, tab4 = st.tabs(["1) Dépenses", "2) Crédits", "3) Hypothèses", "4) Résultats"])

    # ------------------ Tab 1: Dépenses ------------------
    with tab1:
        card_start()
        st.subheader("Dépenses")
        exp = ensure_expense_schema(st.session_state.expenses_df)

        exp_editor = st.data_editor(
            exp,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "nom": st.column_config.TextColumn("Nom", required=True),
                "categorie": st.column_config.TextColumn("Catégorie", required=True),
                "montant": st.column_config.NumberColumn(f"Montant ({CURRENCY})", min_value=0.0, step=10.0, format="%.2f"),
                "frequence": st.column_config.SelectboxColumn("Fréquence", options=FREQ_OPTIONS, required=True),
                "niveau": st.column_config.SelectboxColumn("Essentiel/Confort", options=LEVEL_OPTIONS, required=True),
                "actif": st.column_config.CheckboxColumn("Actif"),
            },
        )
        st.session_state.expenses_df = ensure_expense_schema(exp_editor)

        exp_m, exp_warn = monthlyize_expenses(st.session_state.expenses_df)

        st.markdown("#### Équivalents calculés")
        st.dataframe(
            exp_m[["nom", "categorie", "niveau", "frequence", "montant", "mensuel_equiv", "annuel_equiv", "actif"]],
            use_container_width=True,
        )
        if exp_warn:
            st.warning("Avertissements :\n- " + "\n- ".join(exp_warn))
        card_end()

    # ------------------ Tab 2: Crédits ------------------
    with tab2:
        card_start()
        st.subheader("Crédits / dettes")
        st.write(
            "**Immo** : renseigne **Prix du bien + Apport** (recommandé). "
            "Sinon renseigne **Capital** (capital financé). "
            "Si tu mets aussi un apport sans prix, il sera déduit du capital (warning)."
        )

        loans = ensure_loan_schema(st.session_state.loans_df)
        loans_editor = st.data_editor(
            loans,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "nom": st.column_config.TextColumn("Nom", required=True),
                "type": st.column_config.SelectboxColumn("Type", options=LOAN_TYPE_OPTIONS, required=True),
                "prix_bien": st.column_config.NumberColumn(f"[Immo] Prix du bien ({CURRENCY})", min_value=0.0, step=1000.0, format="%.0f"),
                "apport": st.column_config.NumberColumn(f"[Immo] Apport ({CURRENCY})", min_value=0.0, step=1000.0, format="%.0f"),
                "inclure_frais_notaire": st.column_config.CheckboxColumn("[Immo] Inclure frais de notaire"),
                "frais_notaire_pct": st.column_config.NumberColumn("[Immo] Frais de notaire (%)", min_value=0.0, step=0.1, format="%.1f"),
                "capital": st.column_config.NumberColumn(f"Capital (si pas prix/apport) ({CURRENCY})", min_value=0.0, step=1000.0, format="%.0f"),
                "taux_annuel_pct": st.column_config.NumberColumn("Taux annuel (%)", min_value=0.0, step=0.1, format="%.2f"),
                "duree_annees": st.column_config.NumberColumn("Durée (années)", min_value=0.1, step=0.5, format="%.1f"),
                "assurance_mode": st.column_config.SelectboxColumn("Assurance", options=INS_TYPE_OPTIONS, required=True),
                "assurance_mensuelle": st.column_config.NumberColumn(f"Assurance ({CURRENCY}/mois)", min_value=0.0, step=1.0, format="%.2f"),
                "assurance_taux_annuel_pct": st.column_config.NumberColumn("Assurance (%/an)", min_value=0.0, step=0.05, format="%.2f"),
                "frais": st.column_config.NumberColumn(f"Frais uniques ({CURRENCY})", min_value=0.0, step=10.0, format="%.0f"),
                "etaler_frais": st.column_config.CheckboxColumn("Étaler les frais"),
            },
        )
        st.session_state.loans_df = ensure_loan_schema(loans_editor)

        loans_calc, loans_total, loan_warn = compute_loans(st.session_state.loans_df)

        st.markdown("#### Mensualités calculées")
        if loans_calc.empty:
            st.info("Aucun crédit renseigné.")
        else:
            st.dataframe(
                loans_calc[
                    ["nom", "type", "capital_finance_calc", "taux_annuel_pct", "duree_annees",
                     "mensualite_hors_assurance", "assurance_calc", "frais_mensuels_calc", "mensualite_totale"]
                ],
                use_container_width=True,
            )
            st.metric("Total mensualités crédits", fmt_eur(loans_total))

        if loan_warn:
            st.warning("Avertissements :\n- " + "\n- ".join(loan_warn))

        with st.expander("Option : tableau d’amortissement (pour un prêt)"):
            if not loans_calc.empty:
                pick = st.selectbox("Choisir un prêt", loans_calc["nom"].fillna("(sans nom)").tolist())
                row = loans_calc[loans_calc["nom"] == pick].iloc[0]
                st.dataframe(
                    amortization_schedule(float(row["capital_finance_calc"]), float(row["taux_annuel_pct"]), float(row["duree_annees"])),
                    use_container_width=True,
                )
        card_end()

    # ------------------ Tab 3: Hypothèses ------------------
    with tab3:
        card_start()
        st.subheader("Hypothèses")
        a = st.session_state.assumptions

        c1, c2, c3 = st.columns(3)
        with c1:
            a["pas_rate"] = st.number_input("Taux PAS (ex : 0.081 = 8,1%)", 0.0, 0.99, float(a.get("pas_rate", 0.081)), 0.001, format="%.3f")
        with c2:
            a["ratio_net_brut"] = st.number_input("Ratio net avant IR → brut (approx)", 0.30, 0.95, float(a.get("ratio_net_brut", 0.78)), 0.01, format="%.2f")
        with c3:
            a["marge_imprevus_pct"] = st.number_input("Marge imprévus (0.10 = 10% des dépenses)", 0.0, 1.0, float(a.get("marge_imprevus_pct", 0.10)), 0.01, format="%.2f")

        st.divider()
        d1, d2 = st.columns(2)
        with d1:
            a["epargne_fixe"] = st.number_input(f"Épargne cible ({CURRENCY}/mois)", 0.0, 100000.0, float(a.get("epargne_fixe", 300.0)), 50.0, format="%.0f")
        with d2:
            with st.expander("Avancé (optionnel)"):
                a["epargne_pct_revenu"] = st.number_input("Épargne en % du net après IR (0.10 = 10%)", 0.0, 0.8, float(a.get("epargne_pct_revenu", 0.0)), 0.01, format="%.2f")

        st.divider()
        st.markdown("### Scénario Ambitieux")
        a["amb_bonus_epargne"] = st.number_input(f"Ambitieux : + épargne ({CURRENCY}/mois)", 0.0, 100000.0, float(a.get("amb_bonus_epargne", 200.0)), 50.0, format="%.0f")
        a["amb_bonus_marge"] = st.number_input("Ambitieux : + marge imprévus (ex : 0.05 = +5%)", 0.0, 1.0, float(a.get("amb_bonus_marge", 0.0)), 0.01, format="%.2f")

        st.session_state.assumptions = a
        card_end()

    # ------------------ Tab 4: Résultats (sans graphique) ------------------
    with tab4:
        card_start()
        st.subheader("Résultats (mensuel + annuel)")

        exp_m, exp_warn = monthlyize_expenses(st.session_state.expenses_df)
        _, loans_total, loan_warn = compute_loans(st.session_state.loans_df)
        a = st.session_state.assumptions

        errors = []
        if not (0 <= float(a["pas_rate"]) < 1):
            errors.append("PAS invalide (doit être entre 0 et 0.99).")
        if not (0 < float(a["ratio_net_brut"]) <= 1):
            errors.append("Ratio net→brut invalide (doit être >0 et ≤1).")
        if errors:
            st.error("Erreurs :\n- " + "\n- ".join(errors))
            card_end()
            st.stop()

        warns_all = exp_warn + loan_warn
        if warns_all:
            st.warning("Avertissements :\n- " + "\n- ".join(warns_all))

        scenarios, scen_df = build_scenarios(exp_m, loans_total, a)

        pick = st.radio("Choisir le scénario", scen_df["Scénario"].tolist(), horizontal=True, index=1)
        s = next(x for x in scenarios if x["Scénario"] == pick)

        k1, k2, k3 = st.columns(3)
        k1.metric("Net après IR requis (mensuel)", fmt_eur(s["Net après IR (mensuel)"]))
        k2.metric("Net avant IR requis (mensuel)", fmt_eur(s["Net avant IR (mensuel)"]))
        k3.metric("Brut requis (mensuel)", fmt_eur(s["Brut (mensuel)"]))

        a1, a2, a3 = st.columns(3)
        a1.metric("Net après IR (annuel)", fmt_eur(s["Net après IR (annuel)"]))
        a2.metric("Net avant IR (annuel)", fmt_eur(s["Net avant IR (annuel)"]))
        a3.metric("Brut (annuel)", fmt_eur(s["Brut (annuel)"]))

        st.markdown("#### Décomposition (mensuel)")
        bdf = pd.DataFrame(list(s["_breakdown"].items()), columns=["Poste", "Montant"])
        st.dataframe(bdf, use_container_width=True)

        st.markdown("#### Tableau complet")
        st.dataframe(scen_df, use_container_width=True)

        st.info(
            "Rappel : **Net avant IR** est estimé via le PAS (régularisation possible), "
            "**Brut** est estimé via un ratio net→brut."
        )
        card_end()


if __name__ == "__main__":
    main()
