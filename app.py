"""
app.py — Segmentador de Clientes RFM
ACA Final · Fundamentos de Inteligencia de Negocios (EAD1039) · CUN
Hermes Ariel Campos Villamil · Diana Carolina Cristancho Franco · Yolian Samuel Laguna Mosquera · Elvis Uriel Marciales Chacón · Andrés Felipe Rodríguez Cuesta

Cómo ejecutar:
    streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent
MODELO_DIR   = BASE / "salidas" / "modelo"
FIGURAS_DIR  = BASE / "salidas" / "figuras"

# ── Paleta ───────────────────────────────────────────────────────────────────
AZUL_OSCURO = "#1F4E78"
NARANJA     = "#ED7D31"
AZUL_CLARO  = "#5B9BD5"
VERDE       = "#70AD47"
GRIS        = "#A5A5A5"

AYUDA = {
    "Recencia":        "Días desde la última compra. Valor bajo = cliente más reciente.",
    "Frecuencia":      "Número de transacciones únicas en el período analizado.",
    "Valor_Monetario": "Suma total de compras del cliente en el período.",
    "Ticket_Promedio": "Gasto promedio por transacción (Valor Monetario / Frecuencia).",
}

# ── Página ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Segmentador RFM · CUN",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS + Font Awesome ────────────────────────────────────────────────────────
st.markdown("""
<link rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"
  crossorigin="anonymous"/>

<style>
/* ── Fuente base ── */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #1F4E78 0%, #2E75B6 60%, #5B9BD5 100%);
    padding: 1.6rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.6rem;
    box-shadow: 0 4px 20px rgba(31,78,120,.35);
}
.hero h1 { color: #fff; margin: 0; font-size: 1.7rem; font-weight: 700; }
.hero p  { color: #c8dff2; margin: .35rem 0 0; font-size: .92rem; }

/* ── KPI banner ── */
.kpi-grid { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 130px;
    background: #fff;
    border-radius: 10px;
    padding: .9rem 1.1rem;
    box-shadow: 0 2px 10px rgba(0,0,0,.08);
    border-top: 4px solid #1F4E78;
    text-align: center;
}
.kpi-card .kpi-icon { font-size: 1.5rem; margin-bottom: .3rem; }
.kpi-card .kpi-val  { font-size: 1.6rem; font-weight: 700; color: #1F4E78; }
.kpi-card .kpi-lbl  { font-size: .78rem; color: #777; margin-top: .1rem; }
.kpi-card.orange { border-top-color: #ED7D31; }
.kpi-card.green  { border-top-color: #70AD47; }
.kpi-card.blue   { border-top-color: #5B9BD5; }

/* ── Resultado segmento ── */
.seg-card {
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: 0 3px 14px rgba(0,0,0,.1);
    position: relative;
    overflow: hidden;
}
.seg-card::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 6px; height: 100%;
}
.seg-naranja { background: #fff8f3; }
.seg-naranja::before { background: #ED7D31; }
.seg-azul    { background: #f0f6ff; }
.seg-azul::before    { background: #5B9BD5; }
.seg-verde   { background: #f3faf0; }
.seg-verde::before   { background: #70AD47; }
.seg-gris    { background: #f7f7f7; }
.seg-gris::before    { background: #A5A5A5; }

.seg-badge {
    display: inline-block;
    font-size: .72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .08em;
    padding: .22rem .7rem;
    border-radius: 20px;
    margin-bottom: .5rem;
}
.badge-naranja { background: #ED7D31; color: #fff; }
.badge-azul    { background: #5B9BD5; color: #fff; }
.badge-verde   { background: #70AD47; color: #fff; }
.badge-gris    { background: #A5A5A5; color: #fff; }

.seg-nombre { font-size: 1.4rem; font-weight: 700; color: #1F4E78; }
.seg-sub    { font-size: .92rem; font-weight: 600; color: #555; margin-top: .25rem; }

/* ── Tarjetas de acción ── */
.action-card {
    background: #fff;
    border-radius: 9px;
    padding: .75rem 1rem;
    margin-bottom: .55rem;
    box-shadow: 0 1px 6px rgba(0,0,0,.07);
    display: flex;
    align-items: flex-start;
    gap: .75rem;
    border-left: 3px solid #1F4E78;
}
.action-icon {
    font-size: 1rem;
    color: #1F4E78;
    margin-top: .1rem;
    flex-shrink: 0;
}
.action-text { font-size: .9rem; color: #333; line-height: 1.45; }

/* ── Impacto / advertencia / disclaimer ── */
.impacto {
    background: #e8f4fd;
    border-left: 4px solid #5B9BD5;
    border-radius: 8px;
    padding: .85rem 1rem;
    font-size: .88rem;
    color: #1F4E78;
    margin: .8rem 0;
}
.impacto i { margin-right: .4rem; }

.etico {
    background: #fff8e1;
    border-left: 4px solid #f0ad00;
    border-radius: 8px;
    padding: .85rem 1rem;
    font-size: .88rem;
    color: #6d5700;
    margin: .6rem 0;
}
.etico i { margin-right: .4rem; }

.disclaimer {
    background: #fdf3f0;
    border: 1px solid #f5c6b8;
    border-radius: 8px;
    padding: .9rem 1.1rem;
    font-size: .84rem;
    color: #7a2e1a;
    margin-top: 1rem;
    line-height: 1.55;
}
.disclaimer i { margin-right: .4rem; }

/* ── Sección título ── */
.section-title {
    font-size: 1.1rem; font-weight: 700;
    color: #1F4E78; margin: 1.1rem 0 .6rem;
    display: flex; align-items: center; gap: .5rem;
}
.section-title i { color: #ED7D31; }

/* ── Viz cards ── */
.viz-card {
    background: #fff;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 2px 12px rgba(0,0,0,.07);
}
.viz-label {
    display: inline-block;
    background: #1F4E78;
    color: #fff;
    font-size: .72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .07em;
    padding: .2rem .65rem;
    border-radius: 20px;
    margin-bottom: .5rem;
}
.viz-title { font-size: 1rem; font-weight: 700; color: #1F4E78; margin-bottom: .2rem; }
.viz-caption { font-size: .83rem; color: #666; margin-bottom: .8rem; line-height: 1.5; }

/* ── Ficha técnica ── */
.ficha-card {
    background: #fff;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: .8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,.07);
}
.ficha-card h4 { color: #1F4E78; margin: 0 0 .5rem; font-size: .95rem; }

/* ── Integrante card ── */
.member-grid { display: flex; flex-wrap: wrap; gap: .65rem; margin-top: .5rem; }
.member-chip {
    background: #f0f6ff;
    border: 1px solid #c5daf5;
    border-radius: 20px;
    padding: .35rem .9rem;
    font-size: .85rem;
    color: #1F4E78;
    font-weight: 500;
    display: flex; align-items: center; gap: .4rem;
}
.member-chip i { color: #5B9BD5; }

/* ── Helper ── */
.divider { border: none; border-top: 1px solid #eee; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Carga de artefactos ───────────────────────────────────────────────────────
@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load(MODELO_DIR / "modelo_kmeans.pkl")
    scaler = joblib.load(MODELO_DIR / "scaler.pkl")
    with open(MODELO_DIR / "mapa_segmentos.json", encoding="utf-8") as f:
        meta = json.load(f)
    return modelo, scaler, meta


# ── Recomendaciones por segmento ─────────────────────────────────────────────
def get_recomendacion(nombre_seg: str, centroide: dict) -> dict:
    n      = nombre_seg.lower()
    valor  = centroide.get("Valor_Monetario", 0)
    ticket = centroide.get("Ticket_Promedio",  0)
    frec   = centroide.get("Frecuencia",       0)
    rec    = centroide.get("Recencia",         999)

    if "alto valor" in n and "alta frecuencia" in n:
        return {
            "etiqueta": "Cliente VIP — Retener y Maximizar",
            "color": "naranja", "icono_fa": "fa-star",
            "acciones": [
                ("fa-crown",        f"Invitar al programa de lealtad premium (ticket promedio ${ticket:,.0f})."),
                ("fa-user-tie",     f"Asignar ejecutivo de cuenta: {frec:.0f} compras promedio justifican atención personalizada."),
                ("fa-gift",         "Ofrecer acceso anticipado a nuevas colecciones o servicios exclusivos."),
                ("fa-comment-dots", "Solicitar referencia o testimonio: el historial revela alta satisfacción implícita."),
            ],
            "impacto": f"Este segmento concentra la mayor proporción del ingreso. Reducir la tasa de abandono un 5 % en clientes con valor medio ${valor:,.0f} preserva ingresos significativos sin costo de adquisición.",
            "riesgo":  f"El modelo puede sub-detectar clientes VIP de incorporación reciente. Complementar con revisión manual para recencia < 30 días y ticket ≥ ${ticket*0.8:,.0f}.",
        }
    elif "alto valor" in n:
        return {
            "etiqueta": "Cliente Potencial — Aumentar Frecuencia",
            "color": "azul", "icono_fa": "fa-arrow-trend-up",
            "acciones": [
                ("fa-bell",         f"Campaña de reactivación: {rec:.0f} días sin compra indica ciclo espaciado."),
                ("fa-tags",         f"Ofrecer suscripción o descuento por volumen para pasar de {frec:.0f} a {frec+1:.0f} compras/mes."),
                ("fa-envelope",     "Enviar recordatorio personalizado basado en categorías de su historial."),
                ("fa-box-open",     "Proponer bundle de productos complementarios a sus compras anteriores."),
            ],
            "impacto": f"1 transacción adicional por cliente a ticket ${ticket:,.0f} genera ingreso incremental directo. Capturar el 20 % del segmento duplicaría el retorno frente a adquirir nuevos clientes.",
            "riesgo":  "Evitar contacto excesivo (> 2 comunicaciones/mes): puede deteriorar la percepción de marca y generar opt-out en clientes de alto valor.",
        }
    elif "reciente" in n:
        return {
            "etiqueta": "Cliente Nuevo — Consolidar la Relación",
            "color": "verde", "icono_fa": "fa-seedling",
            "acciones": [
                ("fa-rocket",       f"Activar onboarding: enviar guía de beneficios en los próximos {max(7, int(rec)//4)} días."),
                ("fa-percent",      "Ofrecer incentivo por segunda compra (descuento o envío gratuito)."),
                ("fa-newspaper",    "Incluir en newsletter con contenido educativo y social proof."),
                ("fa-chart-line",   "Monitorear si la segunda compra ocurre en 30 días como KPI de retención."),
            ],
            "impacto": f"Convertir el 20 % de este segmento en compradores recurrentes con ticket ${ticket:,.0f} genera crecimiento neto de ingresos sin costo de adquisición adicional.",
            "riesgo":  "Un cliente reciente con una sola compra puede ser estacional. No activar campañas de recuperación antes de los 60 días de inactividad.",
        }
    else:
        return {
            "etiqueta": "Cliente Dormido — Recuperar o Redirigir",
            "color": "gris", "icono_fa": "fa-moon",
            "acciones": [
                ("fa-rotate-left",  f"Lanzar campaña de recuperación con oferta especial (umbral: {rec:.0f} días sin compra)."),
                ("fa-filter",       "Sub-segmentar: inactivos < 6 meses vs > 12 meses tienen prioridad distinta."),
                ("fa-scale-balanced","Evaluar costo de recuperación vs valor esperado antes de invertir en campañas masivas."),
                ("fa-arrow-right-arrow-left", "Redirigir presupuesto a segmentos de mayor retorno si el costo supera el beneficio."),
            ],
            "impacto": f"Recuperar el 10 % del segmento con ticket ${ticket:,.0f} ofrece retorno incremental. Para inactivos > 12 meses el costo de reactivación suele superar el LTV esperado.",
            "riesgo":  "No inferir capacidad de compra solo por historial: factores externos pueden explicar la inactividad sin implicar abandono permanente.",
        }


# ── Figura explicabilidad ─────────────────────────────────────────────────────
def figura_cliente_vs_centroide(valores_cli: dict, centroide: dict) -> plt.Figure:
    variables = list(valores_cli.keys())
    vals_c = np.array([valores_cli[v] for v in variables], dtype=float)
    vals_s = np.array([centroide[v]   for v in variables], dtype=float)
    maximo = np.maximum(np.abs(vals_c), np.abs(vals_s))
    maximo[maximo == 0] = 1.0
    vals_c_n = vals_c / maximo
    vals_s_n = vals_s / maximo
    etiquetas = [v.replace("_", "\n") for v in variables]
    x = np.arange(len(variables)); w = 0.35
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ax.bar(x - w/2, vals_s_n, w, label="Promedio segmento", color=AZUL_OSCURO, alpha=0.72, edgecolor="none")
    ax.bar(x + w/2, vals_c_n, w, label="Este cliente",      color=NARANJA,     alpha=0.90, edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels(etiquetas, fontsize=8.5)
    ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["Bajo","Medio","Alto"], fontsize=8)
    ax.set_title("Posición relativa vs. promedio del segmento", fontsize=9.5, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2); ax.set_axisbelow(True)
    fig.set_facecolor("white"); ax.set_facecolor("white")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1><i class="fa-solid fa-chart-pie" style="margin-right:.5rem;color:#ED7D31"></i>
      Segmentador de Clientes RFM</h1>
  <p><i class="fa-solid fa-building-columns" style="margin-right:.4rem"></i>
     ACA Final &nbsp;·&nbsp; Fundamentos de Inteligencia de Negocios (EAD1039) &nbsp;·&nbsp; CUN
     &nbsp;|&nbsp; <i class="fa-solid fa-robot" style="margin-right:.3rem"></i>Modelo: K-Means + RFM</p>
</div>
""", unsafe_allow_html=True)


# ── Cargar artefactos ─────────────────────────────────────────────────────────
try:
    modelo, scaler, meta = cargar_artefactos()
    VARIABLES  = meta["variables"]
    SEGMENTOS  = meta["segmentos"]
    CENTROIDES = meta["centroides"]
except FileNotFoundError:
    st.error(
        "No se encontraron los artefactos del modelo. "
        "Ejecuta primero el notebook ACA2_visualizaciones.ipynb completo."
    )
    st.stop()


# ── KPI Banner ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-icon"><i class="fa-solid fa-users" style="color:#1F4E78"></i></div>
    <div class="kpi-val">{meta['clientes_entrenamiento']}</div>
    <div class="kpi-lbl">Clientes analizados</div>
  </div>
  <div class="kpi-card orange">
    <div class="kpi-icon"><i class="fa-solid fa-receipt" style="color:#ED7D31"></i></div>
    <div class="kpi-val">{meta['transacciones_validas']}</div>
    <div class="kpi-lbl">Transacciones válidas</div>
  </div>
  <div class="kpi-card blue">
    <div class="kpi-icon"><i class="fa-solid fa-layer-group" style="color:#5B9BD5"></i></div>
    <div class="kpi-val">{meta['k']}</div>
    <div class="kpi-lbl">Segmentos encontrados</div>
  </div>
  <div class="kpi-card green">
    <div class="kpi-icon"><i class="fa-solid fa-circle-check" style="color:#70AD47"></i></div>
    <div class="kpi-val">{meta['silhouette']:.3f}</div>
    <div class="kpi-lbl">Silhouette score</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "  🔍  Predictor individual  ",
    "  📈  Panorama general  ",
    "  ℹ️  Ficha del modelo  ",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="section-title">
      <i class="fa-solid fa-magnifying-glass-chart"></i> Ingresa los datos del cliente
    </div>
    <p style="color:#555;font-size:.9rem;margin-bottom:1rem;">
      Completa los cuatro indicadores RFM y presiona <strong>Clasificar</strong>.
      El modelo asignará al cliente a uno de los segmentos identificados en el análisis.
    </p>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1.65], gap="large")

    with col_form:
        with st.form("form_prediccion"):
            recencia        = st.number_input("📅  Recencia (días)",       min_value=0,   max_value=3650, value=30,    step=1,    help=AYUDA["Recencia"])
            frecuencia      = st.number_input("🔁  Frecuencia (compras)",  min_value=1,   max_value=500,  value=5,     step=1,    help=AYUDA["Frecuencia"])
            valor_monetario = st.number_input("💰  Valor monetario ($)",   min_value=0.0, max_value=1e8,  value=500.0, step=10.0, help=AYUDA["Valor_Monetario"], format="%.2f")
            ticket_promedio = st.number_input("🧾  Ticket promedio ($)",   min_value=0.0, max_value=1e7,  value=100.0, step=10.0, help=AYUDA["Ticket_Promedio"], format="%.2f")
            enviado = st.form_submit_button("  Clasificar cliente  ", use_container_width=True, type="primary")

        st.markdown("""
        <div style="background:#f0f6ff;border-radius:8px;padding:.8rem 1rem;margin-top:.6rem;font-size:.83rem;color:#1F4E78;">
          <i class="fa-solid fa-circle-info" style="margin-right:.4rem;color:#5B9BD5"></i>
          <strong>¿Qué es RFM?</strong><br>
          <span style="color:#444">
          <b>R</b>ecencia · <b>F</b>recuencia · <b>V</b>alor Monetario — las tres dimensiones clásicas
          para medir el comportamiento de compra y anticipar la respuesta a acciones comerciales.
          </span>
        </div>
        """, unsafe_allow_html=True)

    with col_result:
        if enviado:
            entrada     = np.array([[recencia, frecuencia, valor_monetario, ticket_promedio]])
            entrada_esc = scaler.transform(entrada)
            cluster_id  = int(modelo.predict(entrada_esc)[0])
            nombre_seg  = SEGMENTOS[str(cluster_id)]
            centroide   = CENTROIDES[str(cluster_id)]
            rec_neg     = get_recomendacion(nombre_seg, centroide)
            color       = rec_neg["color"]

            # Segmento card
            st.markdown(f"""
            <div class="seg-card seg-{color}">
              <span class="seg-badge badge-{color}">
                <i class="fa-solid {rec_neg['icono_fa']}" style="margin-right:.3rem"></i>Segmento asignado
              </span>
              <div class="seg-nombre">{nombre_seg}</div>
              <div class="seg-sub">
                <i class="fa-solid fa-bullseye" style="margin-right:.3rem;color:#ED7D31"></i>
                {rec_neg['etiqueta']}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Explicabilidad
            valores_cliente = {
                "Recencia": recencia, "Frecuencia": frecuencia,
                "Valor_Monetario": valor_monetario, "Ticket_Promedio": ticket_promedio,
            }
            with st.expander("  Por qué este segmento — explicabilidad", expanded=True):
                for var in VARIABLES:
                    val_cli = valores_cliente[var]
                    val_cen = centroide[var]
                    rel     = "por encima del" if val_cli >= val_cen else "por debajo del"
                    icono   = "fa-arrow-up" if val_cli >= val_cen else "fa-arrow-down"
                    color_i = "#70AD47" if val_cli >= val_cen else "#ED7D31"
                    st.markdown(
                        f'<div style="font-size:.88rem;margin:.25rem 0;">'
                        f'<i class="fa-solid {icono}" style="color:{color_i};margin-right:.4rem"></i>'
                        f'<b>{var.replace("_"," ")}</b> {val_cli:,.1f} — {rel} promedio del segmento ({val_cen:,.1f})'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                fig_exp = figura_cliente_vs_centroide(valores_cliente, centroide)
                st.pyplot(fig_exp, use_container_width=True)
                plt.close(fig_exp)

            # Recomendaciones
            st.markdown("""
            <div class="section-title" style="margin-top:.8rem">
              <i class="fa-solid fa-lightbulb"></i> Recomendaciones de acción
            </div>
            """, unsafe_allow_html=True)

            for fa_icon, texto in rec_neg["acciones"]:
                st.markdown(f"""
                <div class="action-card">
                  <i class="fa-solid {fa_icon} action-icon"></i>
                  <span class="action-text">{texto}</span>
                </div>
                """, unsafe_allow_html=True)

            # Impacto
            st.markdown(f"""
            <div class="impacto">
              <i class="fa-solid fa-chart-line"></i>
              <strong>Impacto estimado:</strong> {rec_neg['impacto']}
            </div>
            """, unsafe_allow_html=True)

            # Ético
            st.markdown(f"""
            <div class="etico">
              <i class="fa-solid fa-triangle-exclamation"></i>
              <strong>Consideración ética:</strong> {rec_neg['riesgo']}
            </div>
            """, unsafe_allow_html=True)

            # Disclaimer
            st.markdown("""
            <div class="disclaimer">
              <i class="fa-solid fa-shield-halved"></i>
              <strong>Apoyo a la decisión — no criterio definitivo.</strong>
              Este resultado se basa en patrones estadísticos históricos.
              <em>No debe utilizarse como único criterio</em> para negar beneficios,
              excluir clientes ni tomar decisiones de alto impacto sin la revisión de un analista.
              Reentrenamiento sugerido: cada 3 meses.
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:#aaa;">
              <i class="fa-solid fa-magnifying-glass fa-3x" style="margin-bottom:1rem;display:block;color:#c8dff2"></i>
              <p style="font-size:1rem;color:#888;">
                Completa el formulario y presiona <strong style="color:#1F4E78">Clasificar cliente</strong>
                para obtener el segmento y las recomendaciones de acción.
              </p>
            </div>
            """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — PANORAMA GENERAL
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="section-title">
      <i class="fa-solid fa-chart-bar"></i> Análisis general de la base de clientes
    </div>
    <p style="color:#555;font-size:.9rem;margin-bottom:1.2rem;">
      Las tres visualizaciones responden preguntas de negocio distintas aplicando
      atributos pre-atentivos y principios de Gestalt.
      Generadas desde <code>ACA2_visualizaciones.ipynb</code> sobre el dataset real.
    </p>
    """, unsafe_allow_html=True)

    FIGS = [
        ("fig3_comparacion_ingresos.png", "Comparación",
         "Visualización 1 — ¿Qué segmento concentra los ingresos?",
         "Tipo: <b>comparación</b> · Atributo pre-atentivo: color saturado en segmento líder (naranja) · Gestalt: figura-fondo."),
        ("fig4_tendencia_mensual.png", "Tendencia",
         "Visualización 2 — ¿Cómo evoluciona el ingreso en el tiempo?",
         "Tipo: <b>tendencia</b> · Gestalt: continuidad + proximidad · Etiqueta al final de cada serie para evitar leyenda lateral."),
        ("fig5_composicion_clientes.png", "Composición",
         "Visualización 3 — ¿Cómo se distribuyen los clientes?",
         "Tipo: <b>composición</b> · Formato dona: foco en segmento principal sin sesgar por ángulos · Gestalt: región común."),
    ]

    for nombre_fig, tipo, titulo, caption in FIGS:
        ruta = FIGURAS_DIR / nombre_fig
        if not ruta.exists():
            st.warning(f"Figura no encontrada: `{nombre_fig}`. Ejecuta el notebook para generarla.")
            continue
        st.markdown(f"""
        <div class="viz-card">
          <span class="viz-label"><i class="fa-solid fa-chart-bar" style="margin-right:.3rem"></i>{tipo}</span>
          <div class="viz-title">{titulo}</div>
          <div class="viz-caption">{caption}</div>
        """, unsafe_allow_html=True)
        with open(ruta, "rb") as f:
            st.image(f.read(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — FICHA DEL MODELO
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="section-title">
      <i class="fa-solid fa-robot"></i> Ficha técnica del modelo
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algoritmo",            "K-Means")
    c2.metric("Clusters (k)",          meta["k"])
    c3.metric("Silhouette score",      f"{meta['silhouette']:.4f}")
    c4.metric("Entrenado el",          meta["fecha_entrenamiento"])

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        <div class="section-title" style="font-size:.95rem">
          <i class="fa-solid fa-table-columns"></i> Variables de entrada — RFM ampliado
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame([{"Variable": v, "Descripción": AYUDA[v]} for v in VARIABLES]),
            hide_index=True, use_container_width=True,
        )

        st.markdown("""
        <div class="section-title" style="font-size:.95rem;margin-top:1rem">
          <i class="fa-solid fa-layer-group"></i> Segmentos identificados
        </div>
        """, unsafe_allow_html=True)
        iconos_seg = {"naranja": "fa-star", "azul": "fa-arrow-trend-up", "verde": "fa-seedling", "gris": "fa-moon"}
        for idx, nombre in SEGMENTOS.items():
            rec_tmp = get_recomendacion(nombre, CENTROIDES[idx])
            icon    = rec_tmp["icono_fa"]
            st.markdown(
                f'<div style="font-size:.88rem;margin:.3rem 0;">'
                f'<i class="fa-solid {icon}" style="color:#ED7D31;margin-right:.5rem;width:14px"></i>'
                f'<b>Segmento {idx}:</b> {nombre}</div>',
                unsafe_allow_html=True
            )

    with col_b:
        st.markdown("""
        <div class="section-title" style="font-size:.95rem">
          <i class="fa-solid fa-crosshairs"></i> Centroides por segmento (escala original)
        </div>
        """, unsafe_allow_html=True)
        filas = []
        for idx, nombre in SEGMENTOS.items():
            fila = {"Segmento": nombre[:35] + "..." if len(nombre) > 35 else nombre}
            fila.update(CENTROIDES[idx])
            filas.append(fila)
        st.dataframe(pd.DataFrame(filas).round(1), hide_index=True, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title" style="font-size:.95rem">
      <i class="fa-solid fa-triangle-exclamation"></i> Limitaciones y monitoreo futuro
    </div>
    """, unsafe_allow_html=True)

    limitaciones = [
        ("fa-rotate",          "Actualización",        "Reentrenar cada 3 meses o ante cambios en el comportamiento del mercado."),
        ("fa-earth-americas",  "Variables externas",   "No incorpora estacionalidad, canal de compra ni variables socioeconómicas."),
        ("fa-calendar-check",  "Fecha de corte",       f"Recencia calculada contra {meta['fecha_corte_recencia']} — actualizar con cada reentrenamiento."),
        ("fa-scale-balanced",  "Sesgo de datos",       "Si el dataset sub-representa grupos de clientes, el modelo hereda ese sesgo."),
    ]
    for fa, titulo, desc in limitaciones:
        st.markdown(f"""
        <div class="action-card" style="border-left-color:#ED7D31">
          <i class="fa-solid {fa} action-icon" style="color:#ED7D31"></i>
          <span class="action-text"><b>{titulo}:</b> {desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title" style="font-size:.95rem">
      <i class="fa-solid fa-users"></i> Equipo
    </div>
    <div class="member-grid">
      <div class="member-chip"><i class="fa-solid fa-user"></i> Hermes Ariel Campos Villamil</div>
      <div class="member-chip"><i class="fa-solid fa-user"></i> Diana Carolina Cristancho Franco</div>
      <div class="member-chip"><i class="fa-solid fa-user"></i> Yolian Samuel Laguna Mosquera</div>
      <div class="member-chip"><i class="fa-solid fa-user"></i> Elvis Uriel Marciales Chacón</div>
      <div class="member-chip"><i class="fa-solid fa-user"></i> Andrés Felipe Rodríguez Cuesta</div>
    </div>
    <p style="font-size:.8rem;color:#999;margin-top:.8rem">
      CUN · Especialización en Analítica Avanzada de Datos · Fundamentos de Inteligencia de Negocios (EAD1039)
    </p>
    """, unsafe_allow_html=True)
