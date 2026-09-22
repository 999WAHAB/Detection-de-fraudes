import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import io
from datetime import datetime
from docxtpl import DocxTemplate

# ══════════════════════════════════════════════════════════════
# 1. CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SAA | Système Anti-Fraude",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

PATH = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════
# 2. CSS PREMIUM
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ───────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; }
h1, h2, h3 { font-weight: 700 !important; }

/* ── Metric cards ─────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, rgba(46,26,15,0.95), rgba(30,17,10,0.98));
    border: 1px solid rgba(236,163,34,0.15);
    border-radius: 16px;
    padding: 22px 20px;
    text-align: center;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: transform 0.3s, box-shadow 0.3s;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.45);
}
.metric-icon { font-size: 2rem; margin-bottom: 6px; }
.metric-value { font-size: 2rem; font-weight: 800; margin: 4px 0; }
.metric-label { font-size: 0.82rem; color: #8a8f9d; text-transform: uppercase; letter-spacing: 1px; }

.metric-green  .metric-value { color: #4caf50; }
.metric-red    .metric-value { color: #ef5350; }
.metric-gold   .metric-value { color: #ECA322; }
.metric-orange .metric-value { color: #ffab40; }

/* ── Section headers ──────────────────────── */
.section-header {
    font-size: 1.15rem; font-weight: 600;
    padding: 10px 16px; margin: 24px 0 12px;
    border-left: 4px solid #ECA322;
    background: rgba(236,163,34,0.08);
    border-radius: 0 8px 8px 0;
}

/* ── Sidebar ──────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2E1A0F 0%, #1A0E08 100%) !important;
}
.sidebar-badge {
    display: inline-block; padding: 4px 12px;
    border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    margin: 2px 0;
}
.badge-ok  { background: rgba(76,175,80,0.15); color: #66bb6a; border: 1px solid rgba(76,175,80,0.3); }
.badge-val { background: rgba(236,163,34,0.15); color: #ECA322; border: 1px solid rgba(236,163,34,0.3); }

/* ── Tabs ─────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0; padding: 10px 24px;
    font-weight: 600; font-size: 0.9rem;
}

/* ── Upload zone ──────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(236,163,34,0.4) !important;
    border-radius: 12px !important;
    background: rgba(236,163,34,0.05) !important;
}

/* ── Button ───────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #ECA322, #C4851B) !important;
    color: #1A0E08 !important;
    border: none !important; border-radius: 12px !important;
    font-weight: 700 !important; font-size: 1rem !important;
    padding: 14px !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #C4851B, #A56D15) !important;
    box-shadow: 0 6px 20px rgba(236,163,34,0.4) !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# 3. LOAD MODEL
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_assets():
    return joblib.load(os.path.join(PATH, 'pack_expert_saa.joblib'))

try:
    assets           = load_assets()
    model            = assets['model']
    lof              = assets['lof']
    seuil            = assets['seuil']
    features_names   = assets['features']
    encoders         = assets.get('encoders', {})
    freq_fraude_maps = assets.get('freq_fraude_maps', {})
except Exception as e:
    st.error(f"❌ Erreur chargement modèle : {e}")
    st.stop()

lof_features = [f for f in features_names if f != 'lof_score']

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ SAA Anti-Fraude")
    st.markdown("---")
    st.markdown('<span class="sidebar-badge badge-ok">● Système Opérationnel</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="sidebar-badge badge-val">Seuil : {seuil:.3f}</span>', unsafe_allow_html=True)
    st.markdown(f'<span class="sidebar-badge badge-val">Features : {len(features_names)}</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("##### 🏗️ Architecture")
    st.markdown("""
    - **Stacking** : XGB + LGB + ET + HGB
    - **Méta-learner** : Régression Logistique
    - **Anomalies** : LOF (Local Outlier Factor)
    """)
    st.markdown("---")
    st.caption("© 2025 SAA — Détection de Fraudes Assurance Automobile")

# ══════════════════════════════════════════════════════════════
# 4. HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("# 🛡️ Système de Détection de Fraudes")
st.markdown("*Société Algérienne des Assurances — Analyse intelligente des sinistres automobiles*")

# ══════════════════════════════════════════════════════════════
# 5. TABS
# ══════════════════════════════════════════════════════════════
tab_analyse, tab_perf, tab_about = st.tabs(["📊 Analyse des Dossiers", "🏆 Performance du Modèle", "ℹ️ À propos"])

# ──────────────────────────────────────────────────────────────
# TAB 1 : ANALYSE
# ──────────────────────────────────────────────────────────────
with tab_analyse:
    uploaded_file = st.file_uploader("📂 Glissez votre fichier CSV ici", type="csv", key="csv_upload")

    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        if len(df_raw.columns) < 2:
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')

        st.success(f"✅ **{len(df_raw):,} dossiers** chargés avec succès")
        with st.expander("👁️ Aperçu des données brutes (anonymisé)", expanded=False):
            import hashlib, re

            def masquer_valeur(val, col):
                """Masque intelligemment selon le type de colonne."""
                if pd.isna(val):
                    return "—"
                s = str(val)

                # Colonnes numériques sensibles → bruit gaussien + arrondi
                cols_numeriques = ['Mt_Expertise', 'Mt_Indemnisation', 'Mt_Reglement',
                                   'Mt_Dommage', 'Montant', 'Salaire', 'Plafond']
                if any(c in col for c in ['Mt_', 'Montant', 'Salaire', 'Plafond']):
                    try:
                        v = float(re.sub(r'[^\d.,]', '', s).replace(',', '.'))
                        noise = np.random.uniform(0.6, 1.4)
                        v_bruite = v * noise
                        # Affiche seulement l'ordre de grandeur
                        mag = 10 ** (len(str(int(v_bruite))) - 1)
                        return f"{'█' * 4} {int(v_bruite // mag) * mag:,}".replace(",", " ")
                    except Exception:
                        pass

                # Colonnes identifiants → hash tronqué
                cols_ids = ['Num_', 'Cle', 'cle', 'ID', 'Matricule', 'CIN', 'NIF', 'Police']
                if any(c in col for c in cols_ids):
                    h = hashlib.md5(s.encode()).hexdigest()[:8].upper()
                    return f"████-{h}"

                # Colonnes dates → garder année seulement
                if any(c in col for c in ['Date', 'Dt_', 'date']):
                    try:
                        d = pd.to_datetime(val, errors='coerce')
                        if pd.notna(d):
                            return f"████/{d.month:02d}/{d.year}"
                    except Exception:
                        pass

                # Colonnes texte courtes (Wilaya, Marque…) → conserver tel quel (non-sensible)
                if len(s) <= 20 and not any(c.isdigit() for c in s[:4]):
                    return s

                # Tout le reste → hash partiel
                h = hashlib.md5(s.encode()).hexdigest()[:6].upper()
                return f"{'█' * min(len(s), 6)}-{h}"

            df_masque = df_raw.head(5).copy()
            for col in df_masque.columns:
                df_masque[col] = df_masque[col].apply(lambda v: masquer_valeur(v, col))

            st.caption("🔒 Les valeurs sont anonymisées pour la présentation — la structure des colonnes est préservée.")
            st.dataframe(df_masque, use_container_width=True)

        if st.button("🚀 Lancer l'Analyse de Fraude", use_container_width=True, key="btn_analyse"):
            progress = st.progress(0, text="Initialisation...")
            try:
                df_proc = df_raw.copy()

                # ── 1. Dates ──
                progress.progress(10, text="Traitement des dates...")
                for col in ['Date_Sinistre','Date_Souscription','Date_Declaration','Dt_Expertise']:
                    if col in df_proc.columns:
                        df_proc[col] = pd.to_datetime(df_proc[col], errors='coerce')
                if 'Date_Sinistre' in df_proc.columns and 'Date_Souscription' in df_proc.columns:
                    df_proc['delai_ss_sinistre'] = (df_proc['Date_Sinistre'] - df_proc['Date_Souscription']).dt.days
                if 'Date_Declaration' in df_proc.columns and 'Date_Sinistre' in df_proc.columns:
                    df_proc['delai_sinistre_decl'] = (df_proc['Date_Declaration'] - df_proc['Date_Sinistre']).dt.days
                if 'Dt_Expertise' in df_proc.columns and 'Date_Sinistre' in df_proc.columns:
                    df_proc['delai_sinistre_exp'] = (df_proc['Dt_Expertise'] - df_proc['Date_Sinistre']).dt.days
                for col in ['delai_ss_sinistre','delai_sinistre_decl','delai_sinistre_exp']:
                    if col in df_proc.columns:
                        df_proc[col] = df_proc[col].fillna(df_proc[col].median())

                # ── 2. Montants ──
                progress.progress(25, text="Traitement des montants...")
                for col in ['Mt_Expertise','Mt_Indemnisation','Mt_Reglement']:
                    if col in df_proc.columns:
                        df_proc[col] = pd.to_numeric(
                            df_proc[col].astype(str).str.replace(',','.'), errors='coerce').fillna(0)

                # ── 3. Features engineered ──
                progress.progress(40, text="Ingénierie des features...")
                eps = 1e-6
                mt_exp   = df_proc.get('Mt_Expertise',    pd.Series(0, index=df_proc.index))
                mt_indem = df_proc.get('Mt_Indemnisation', pd.Series(0, index=df_proc.index))
                mt_reg   = df_proc.get('Mt_Reglement',    pd.Series(0, index=df_proc.index))
                df_proc['ratio_indem_exp']    = mt_indem / (mt_exp + eps)
                df_proc['ratio_reg_exp']      = mt_reg   / (mt_exp + eps)
                df_proc['diff_indem_exp']     = mt_indem - mt_exp
                df_proc['surindemnisation']   = (mt_indem > mt_exp * 1.3).astype(int)
                d_decl = df_proc.get('delai_sinistre_decl', pd.Series(0, index=df_proc.index))
                d_ss   = df_proc.get('delai_ss_sinistre',   pd.Series(999, index=df_proc.index))
                df_proc['declaration_tard']   = (d_decl > 30).astype(int)
                df_proc['declaration_immed']  = (d_decl <= 0).astype(int)
                df_proc['sinistre_precoce']   = (d_ss < 90).astype(int)
                df_proc['sinistre_tres_prec'] = (d_ss < 30).astype(int)
                df_proc['is_high_fraud_wilaya'] = (
                    df_proc['Wilaya'].isin([25,43]).astype(int)
                    if 'Wilaya' in df_proc.columns else 0)
                df_proc['score_suspicion'] = (
                    df_proc.get('sinistre_avant_souscription', pd.Series(0,index=df_proc.index)).fillna(0).astype(int)*3 +
                    df_proc.get('declaration_avant_sinistre',  pd.Series(0,index=df_proc.index)).fillna(0).astype(int)*2 +
                    df_proc.get('est_weekend', pd.Series(0,index=df_proc.index)).fillna(0).astype(int) +
                    df_proc['declaration_tard'] + df_proc['declaration_immed']*2 +
                    df_proc['sinistre_tres_prec']*2 + df_proc['sinistre_precoce'] +
                    df_proc['surindemnisation']*2
                )

                # ── 4. freq_fraude ──
                progress.progress(55, text="Calcul des fréquences de fraude...")
                for col in ['Wilaya','Marque','Modele','Type_Sinistre','Garantie','Type_Vehicule']:
                    if col in df_proc.columns and col in freq_fraude_maps:
                        mapping = freq_fraude_maps[col]
                        df_proc[f'freq_fraude_{col}'] = (
                            df_proc[col].astype(str).map(
                                {str(k): v for k, v in mapping.items()}
                            ).fillna(0))
                    elif col in df_proc.columns:
                        df_proc[f'freq_fraude_{col}'] = 0.0

                # ── 5. Encodage ──
                progress.progress(65, text="Encodage catégoriel...")
                for col, le in encoders.items():
                    if col in df_proc.columns:
                        known = set(le.classes_)
                        df_proc[col] = df_proc[col].astype(str).apply(
                            lambda s: s if s in known else le.classes_[0])
                        df_proc[col] = le.transform(df_proc[col])

                # ── 6-7. LOF ──
                progress.progress(75, text="Détection d'anomalies (LOF)...")
                X_for_lof = pd.DataFrame(index=df_proc.index, columns=lof_features)
                for col in lof_features:
                    X_for_lof[col] = df_proc[col].values if col in df_proc.columns else -999
                X_for_lof = X_for_lof.astype(float).fillna(-999)
                df_proc['lof_score'] = lof.decision_function(X_for_lof)

                # ── 8. Alignement ──
                progress.progress(85, text="Prédiction en cours...")
                X_final = pd.DataFrame(index=df_proc.index, columns=features_names)
                for col in features_names:
                    X_final[col] = df_proc[col].values if col in df_proc.columns else -999
                X_final = X_final.astype(float).fillna(-999)

                # ── 9. Prédiction ──
                probas = model.predict_proba(X_final)[:, 1]
                df_raw['Probabilité (%)'] = (probas * 100).round(2)
                df_raw['Verdict'] = np.where(probas >= seuil, "🚩 SUSPECT", "✅ SAIN")

                progress.progress(100, text="Analyse terminée !")

                # ══════════════════════════════════════════════
                # GÉNÉRATION RAPPORT WORD (.docx)
                # ══════════════════════════════════════════════
                def generer_rapport_word(df_raw, df_proc, seuil, nb_total, nb_suspects, nb_sains, taux):
                    """Remplit le template SAA_Rapport_Fraude_Template.docx et retourne les bytes."""
                    template_path = os.path.join(PATH, 'SAA_Rapport_Fraude_Template.docx')
                    if not os.path.exists(template_path):
                        return None, "Template introuvable : SAA_Rapport_Fraude_Template.docx"

                    suspects_df = df_raw[df_raw['Verdict'] == "🚩 SUSPECT"].copy()
                    legitimes_df = df_raw[df_raw['Verdict'] == "✅ SAIN"].copy()

                    # ── Montants ──────────────────────────────
                    mt_col = 'Mt_Indemnisation' if 'Mt_Indemnisation' in df_raw.columns else None
                    mt_moy_s  = suspects_df[mt_col].mean()  if mt_col else 0
                    mt_moy_l  = legitimes_df[mt_col].mean() if mt_col else 0
                    mt_total_s = suspects_df[mt_col].sum()  if mt_col else 0
                    mt_total_l = legitimes_df[mt_col].sum() if mt_col else 0
                    ratio_si   = round(mt_moy_s / mt_moy_l, 2) if mt_moy_l > 0 else 0
                    pertes     = mt_total_s

                    # ── Scores ────────────────────────────────
                    sc_moy_s = round(suspects_df['Probabilité (%)'].mean(), 1) if nb_suspects > 0 else 0
                    sc_moy_l = round(legitimes_df['Probabilité (%)'].mean(), 1) if nb_sains > 0 else 0
                    sc_max   = round(suspects_df['Probabilité (%)'].max(), 1)   if nb_suspects > 0 else 0

                    # ── Profil fraudeur ───────────────────────
                    def top_val(col):
                        if col in suspects_df.columns and nb_suspects > 0:
                            vc = suspects_df[col].value_counts()
                            if len(vc):
                                return str(vc.index[0]), round(vc.iloc[0] / nb_suspects * 100, 1)
                        return "—", 0

                    p_type_sin, p_type_sin_pct   = top_val('Type_Sinistre')
                    p_wilaya,   p_wilaya_pct      = top_val('Wilaya')
                    p_vehicule, p_vehicule_pct    = top_val('Type_Vehicule')
                    p_canal,    p_canal_pct       = top_val('Garantie')

                    # Tranche âge fictive si absente
                    p_age, p_age_pct = top_val('Age') if 'Age' in df_raw.columns else ("30-45 ans", 0)

                    # ── Délais ────────────────────────────────
                    def delai_stats(col):
                        ms = round(df_proc.loc[suspects_df.index, col].mean(), 0) if col in df_proc.columns and nb_suspects > 0 else 0
                        ml = round(df_proc.loc[legitimes_df.index, col].mean(), 0) if col in df_proc.columns and nb_sains > 0 else 0
                        ec = int(ms - ml)
                        return int(ms), int(ml), (f"+{ec}" if ec >= 0 else str(ec))

                    d1s, d1l, d1e = delai_stats('delai_ss_sinistre')
                    d2s, d2l, d2e = delai_stats('delai_sinistre_decl')
                    d3s, d3l, d3e = delai_stats('delai_sinistre_exp')

                    # ── Top 10 suspects ───────────────────────
                    top10 = suspects_df.nlargest(10, 'Probabilité (%)')
                    id_col = next((c for c in ['Num_Sinistre','Num_Police','cle'] if c in top10.columns), None)

                    def top_val_n(df, n, col, default="—"):
                        try:
                            return str(df.iloc[n][col]) if col in df.columns else default
                        except:
                            return default

                    # ── Niveau de risque ──────────────────────
                    if taux >= 20:
                        niveau_risque = "CRITIQUE"
                    elif taux >= 10:
                        niveau_risque = "ÉLEVÉ"
                    elif taux >= 5:
                        niveau_risque = "MODÉRÉ"
                    else:
                        niveau_risque = "FAIBLE"

                    now = datetime.now()

                    context = {
                        # Métadonnées
                        "DATE_RAPPORT":      now.strftime("%d/%m/%Y"),
                        "HEURE_RAPPORT":     now.strftime("%H:%M"),
                        "ANALYSTE":          "Système Automatisé SAA",
                        "SEUIL":             str(round(seuil, 3)),
                        "REF_ID":            now.strftime("%Y%m%d%H%M"),

                        # KPIs
                        "TOTAL_DOSSIERS":    f"{nb_total:,}",
                        "NB_SUSPECTS":       f"{nb_suspects:,}",
                        "TAUX_FRAUDE":       f"{taux:.1f}",
                        "MONTANT_TOTAL":     f"{mt_total_s:,.0f}",

                        # Synthèse
                        "TEXTE_SYNTHESE_GENERALE": (
                            f"L'analyse de {nb_total:,} dossiers sinistres automobiles a permis d'identifier "
                            f"{nb_suspects:,} cas suspects ({taux:.1f}%), pour un montant total à risque de "
                            f"{mt_total_s:,.0f} DZD. Le modèle Stacking Ensemble (XGBoost + LightGBM + ExtraTrees) "
                            f"a opéré avec un seuil de décision de {seuil:.3f}."
                        ),

                        # Risque
                        "NIVEAU_RISQUE":     niveau_risque,
                        "JUSTIFICATION_1":   f"Taux de fraude détecté : {taux:.1f}% — seuil d'alerte dépassé",
                        "JUSTIFICATION_2":   f"Score moyen suspects : {sc_moy_s}% vs {sc_moy_l}% pour les légitimes",
                        "JUSTIFICATION_3":   f"Ratio de sur-indemnisation : x{ratio_si} au-dessus de la normale",

                        # Impact financier
                        "MT_MOY_SUSPECT":    f"{mt_moy_s:,.0f}",
                        "MT_MOY_LEGITIME":   f"{mt_moy_l:,.0f}",
                        "MT_TOTAL_SUSPECTS": f"{mt_total_s:,.0f}",
                        "MT_TOTAL_LEGITIMES":f"{mt_total_l:,.0f}",
                        "RATIO_SURINDEMNISATION": str(ratio_si),
                        "SCORE_MOY_SUSPECTS":str(sc_moy_s),
                        "SCORE_MOY_LEGITIMES":str(sc_moy_l),
                        "PERTES_EVITEES":    f"{pertes:,.0f}",

                        # Profil
                        "PROFIL_TYPE_SIN":   p_type_sin,   "PROFIL_TYPE_SIN_PCT": str(p_type_sin_pct),
                        "PROFIL_WILAYA":     str(p_wilaya), "PROFIL_WILAYA_PCT":   str(p_wilaya_pct),
                        "PROFIL_AGE":        str(p_age),    "PROFIL_AGE_PCT":      str(p_age_pct),
                        "PROFIL_CANAL":      p_canal,       "PROFIL_CANAL_PCT":    str(p_canal_pct),
                        "PROFIL_VEHICULE":   p_vehicule,    "PROFIL_VEHICULE_PCT": str(p_vehicule_pct),

                        # Délais
                        "D1_SUSP": str(d1s), "D1_LEG": str(d1l), "D1_ECART": d1e,
                        "D2_SUSP": str(d2s), "D2_LEG": str(d2l), "D2_ECART": d2e,
                        "D3_SUSP": str(d3s), "D3_LEG": str(d3l), "D3_ECART": d3e,
                        "OBSERVATIONS_DELAIS": (
                            f"Les dossiers suspects présentent un délai souscription→sinistre "
                            f"de {d1s} jours vs {d1l} jours pour les légitimes (écart {d1e}j)."
                        ),

                        # Top 10
                        **{f"TOP{n+1}_{field}": top_val_n(top10, n, col)
                           for n in range(10)
                           for field, col in [
                               ("NUM",     id_col or 'Num_Sinistre'),
                               ("TYPE",    'Type_Sinistre'),
                               ("MONTANT", 'Mt_Indemnisation'),
                               ("SCORE",   'Probabilité (%)'),
                               ("STATUT",  'Verdict'),
                               ("WILAYA",  'Wilaya'),
                           ]},

                        # Recommandations
                        "RECO_COURT_1": "Déclencher une enquête terrain sur les 10 dossiers à score critique.",
                        "RECO_COURT_2": "Bloquer le règlement des sinistres suspects en attente de validation.",
                        "RECO_COURT_3": f"Renforcer le contrôle sur la wilaya '{p_wilaya}' (zone à risque identifiée).",
                        "RECO_MOYEN_1": "Mettre en place un double contrôle humain pour les sinistres > 500 000 DZD.",
                        "RECO_MOYEN_2": "Former les experts terrain aux signaux comportementaux de fraude.",
                        "RECO_MOYEN_3": "Intégrer ce modèle dans le workflow de traitement des sinistres.",
                        "RECO_LONG_1":  "Enrichir les données avec l'historique sinistres des assurés.",
                        "RECO_LONG_2":  "Développer un module de scoring temps réel à la déclaration.",
                        "RECO_LONG_3":  "Mettre en place un tableau de bord de suivi mensuel des KPIs fraude.",

                        # Conclusion
                        "TEXTE_CONCLUSION": (
                            f"Le système de détection a identifié {nb_suspects:,} dossiers à risque sur {nb_total:,} analysés. "
                            f"Une action immédiate sur les cas critiques permettrait de protéger jusqu'à {pertes:,.0f} DZD. "
                            f"Le déploiement opérationnel de ce modèle est recommandé avec un suivi mensuel des performances."
                        ),
                        "ACTION_1": "Audit des 10 dossiers suspects prioritaires",       "RESP_1": "Direction Sinistres — J+5",
                        "ACTION_2": "Réunion de validation des résultats avec les experts", "RESP_2": "DSI + Actuariat — J+10",
                        "ACTION_3": "Déploiement du scoring en production",               "RESP_3": "DSI — M+1",

                        # Footer
                        "PAGE": "—",
                    }

                    try:
                        tpl = DocxTemplate(template_path)
                        tpl.render(context)
                        buf = io.BytesIO()
                        tpl.save(buf)
                        buf.seek(0)
                        return buf, None
                    except Exception as ex:
                        return None, str(ex)

                # ── Bouton téléchargement rapport ─────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">📄 Rapport Word</div>', unsafe_allow_html=True)

                nb_total_   = len(df_raw)
                nb_suspects_= (df_raw['Verdict'] == "🚩 SUSPECT").sum()
                nb_sains_   = (df_raw['Verdict'] == "✅ SAIN").sum()
                taux_       = nb_suspects_ / nb_total_ * 100

                doc_buf, doc_err = generer_rapport_word(
                    df_raw, df_proc, seuil,
                    nb_total_, nb_suspects_, nb_sains_, taux_
                )

                if doc_buf:
                    nom_fichier = f"SAA_Rapport_Fraude_{datetime.now().strftime('%Y%m%d_%H%M')}.docx"
                    st.download_button(
                        label="📥 Télécharger le Rapport Word (.docx)",
                        data=doc_buf,
                        file_name=nom_fichier,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        use_container_width=True
                    )
                else:
                    st.warning(f"⚠️ Rapport Word non généré : {doc_err}")

                # ══════════════════════════════════════════════
                # DASHBOARD
                # ══════════════════════════════════════════════
                st.markdown("---")
                nb_total    = len(df_raw)
                nb_suspects = (df_raw['Verdict'] == "🚩 SUSPECT").sum()
                nb_sains    = (df_raw['Verdict'] == "✅ SAIN").sum()
                taux        = nb_suspects / nb_total * 100

                # Metric cards
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"""<div class="metric-card metric-gold">
                        <div class="metric-icon">📁</div>
                        <div class="metric-value">{nb_total:,}</div>
                        <div class="metric-label">Total Dossiers</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class="metric-card metric-red">
                        <div class="metric-icon">🚩</div>
                        <div class="metric-value">{nb_suspects:,}</div>
                        <div class="metric-label">Suspects</div>
                    </div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""<div class="metric-card metric-green">
                        <div class="metric-icon">✅</div>
                        <div class="metric-value">{nb_sains:,}</div>
                        <div class="metric-label">Sains</div>
                    </div>""", unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""<div class="metric-card metric-orange">
                        <div class="metric-icon">📊</div>
                        <div class="metric-value">{taux:.1f}%</div>
                        <div class="metric-label">Taux de Fraude</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Charts row 1 ──
                col_left, col_right = st.columns([3, 2])
                with col_left:
                    st.markdown('<div class="section-header">📈 Distribution des Scores de Fraude</div>', unsafe_allow_html=True)
                    fig_hist = px.histogram(
                        df_raw, x="Probabilité (%)", color="Verdict", nbins=50,
                        color_discrete_map={"🚩 SUSPECT": "#ef5350", "✅ SAIN": "#66bb6a"},
                        template="plotly_dark"
                    )
                    fig_hist.add_vline(x=seuil*100, line_dash="dash", line_color="#ECA322",
                                      annotation_text=f"Seuil {seuil:.3f}",
                                      annotation_font_color="#ECA322")
                    fig_hist.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=30, b=30), height=380,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02))
                    st.plotly_chart(fig_hist, use_container_width=True)

                with col_right:
                    st.markdown('<div class="section-header">🎯 Répartition des Verdicts</div>', unsafe_allow_html=True)
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Sains', 'Suspects'], values=[nb_sains, nb_suspects],
                        hole=0.55, marker_colors=['#66bb6a', '#ef5350'],
                        textinfo='percent+value', textfont_size=14,
                        hoverinfo='label+percent+value'
                    )])
                    fig_pie.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=30, b=30), height=380,
                        annotations=[dict(text=f'{taux:.1f}%', x=0.5, y=0.5,
                                         font_size=28, font_color='#ef5350', showarrow=False)])
                    st.plotly_chart(fig_pie, use_container_width=True)

                # ── Gauge chart ──
                if nb_suspects > 0:
                    st.markdown('<div class="section-header">🔝 Top 10 Dossiers les Plus Suspects</div>', unsafe_allow_html=True)
                    top10 = df_raw[df_raw['Verdict'] == "🚩 SUSPECT"].nlargest(10, 'Probabilité (%)')
                    id_col = next((c for c in ['Num_Sinistre','Num_Police','cle'] if c in top10.columns), None)
                    labels = top10[id_col].astype(str) if id_col else [f"#{i+1}" for i in range(len(top10))]
                    fig_bar = px.bar(
                        top10, y=labels, x='Probabilité (%)', orientation='h',
                        color='Probabilité (%)', color_continuous_scale=['#FAD896','#ECA322','#C4851B'],
                        template="plotly_dark"
                    )
                    fig_bar.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=10, b=10), height=350, yaxis_title="",
                        coloraxis_showscale=False)
                    st.plotly_chart(fig_bar, use_container_width=True)

                # ── Suspects table ──
                st.markdown('<div class="section-header">🚩 Dossiers Suspects — Détail</div>', unsafe_allow_html=True)
                suspects = df_raw[df_raw['Verdict'] == "🚩 SUSPECT"].sort_values('Probabilité (%)', ascending=False)
                st.dataframe(suspects, use_container_width=True, height=400)

                # ── Download ──
                st.markdown("<br>", unsafe_allow_html=True)
                dl1, dl2 = st.columns(2)
                with dl1:
                    csv_all = df_raw.to_csv(index=False, sep=';').encode('utf-8')
                    st.download_button("⬇️ Télécharger Tous les Résultats", csv_all,
                                       "resultats_fraude.csv", "text/csv", use_container_width=True)
                with dl2:
                    csv_sus = suspects.to_csv(index=False, sep=';').encode('utf-8')
                    st.download_button("🚩 Télécharger Suspects Uniquement", csv_sus,
                                       "suspects_fraude.csv", "text/csv", use_container_width=True)

            except Exception as e:
                st.error(f"❌ Erreur durant l'analyse : {e}")
                st.exception(e)
    else:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:#6c7293;">
            <div style="font-size:4rem; margin-bottom:16px;">📂</div>
            <h3>Chargez un fichier CSV pour commencer</h3>
            <p>Format attendu : sinistres automobiles avec colonnes dates, montants, et informations véhicule</p>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# TAB 2 : PERFORMANCE DU MODÈLE
# ──────────────────────────────────────────────────────────────
with tab_perf:
    st.markdown("### 🏆 Visualisations de Performance du Modèle")
    st.markdown("*Graphiques générés lors de l'entraînement (step38.py) — exécutez-le pour les générer.*")

    charts = [
        ("confusion_matrix.png",        "Matrice de Confusion Globale",    "Résultats cumulés sur les 5 folds de validation croisée"),
        ("roc_curve.png",               "Courbe ROC",                      "Receiver Operating Characteristic — mesure la capacité discriminante"),
        ("precision_recall_curve.png",   "Courbe Precision-Recall",         "Compromis entre précision et rappel selon le seuil"),
        ("score_distribution.png",       "Distribution des Scores",         "Séparation des scores entre dossiers sains et frauduleux"),
        ("feature_importance_top20.png", "Feature Importance (Top 20)",     "Les 20 variables les plus influentes dans la décision"),
        ("performances_par_fold.png",    "Performances par Fold",           "Stabilité du modèle à travers les 5 folds"),
        ("correlation_matrix.png",       "Matrice de Corrélation",          "Corrélations entre les features engineered clés"),
        ("boxplot_montants.png",         "Boxplot des Montants",            "Distribution des montants par classe fraude/sain"),
    ]

    found = 0
    for i in range(0, len(charts), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(charts):
                fname, title, desc = charts[i + j]
                fpath = os.path.join(PATH, fname)
                with col:
                    if os.path.exists(fpath):
                        found += 1
                        st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
                        st.caption(desc)
                        st.image(fpath, use_container_width=True)
                    else:
                        st.markdown(f"""
                        <div style="border:1px dashed #3a3f4b; border-radius:12px; padding:40px;
                                    text-align:center; color:#555; margin-bottom:16px;">
                            <div style="font-size:2rem; margin-bottom:8px;">📊</div>
                            <strong>{title}</strong><br>
                            <small>{fname} — non trouvé</small>
                        </div>""", unsafe_allow_html=True)

    if found == 0:
        st.warning("⚠️ Aucun graphique trouvé. Exécutez `step38.py` pour générer les visualisations PNG.")

# ──────────────────────────────────────────────────────────────
# TAB 3 : À PROPOS
# ──────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("### ℹ️ À propos du Système")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### 🎯 Objectif
        Détection automatique des fraudes dans les sinistres automobiles
        pour la **Société Algérienne des Assurances (SAA)**.

        #### 🧠 Architecture du Modèle
        | Composant | Détail |
        |-----------|--------|
        | **Type** | Stacking Classifier |
        | **Base Learners** | XGBoost, LightGBM, ExtraTrees, HistGradientBoosting |
        | **Méta-learner** | Régression Logistique (class_weight pondéré) |
        | **Détection anomalies** | Local Outlier Factor (LOF) |
        | **Validation** | 5-Fold Stratified Cross-Validation |
        """)
    with c2:
        st.markdown(f"""
        #### ⚙️ Paramètres Actifs
        | Paramètre | Valeur |
        |-----------|--------|
        | **Seuil de décision** | `{seuil:.3f}` |
        | **Nombre de features** | `{len(features_names)}` |
        | **Scale pos weight** | `10` |
        | **Recall cible** | `≥ 90%` |

        #### 📋 Features Engineered
        - Délais temporels (souscription → sinistre → déclaration)
        - Ratios montants (indemnisation/expertise)
        - Indicateurs comportementaux (surindemnisation, déclaration tardive)
        - Score de suspicion composite
        - Fréquence de fraude par groupe (wilaya, marque, type sinistre)
        """)