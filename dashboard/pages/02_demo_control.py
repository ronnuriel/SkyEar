import streamlit as st


st.set_page_config(page_title="SkyEar Demo Control", layout="wide")
st.title("SkyEar Demo Control")
st.caption("Passive visualization demo commands and operator expectations.")

st.subheader("Run commands")
st.markdown("Terminal 1:")
st.code("PYTHONPATH=. uvicorn server.api:app --reload --host 0.0.0.0 --port 8080", language="bash")

st.markdown("Terminal 2:")
st.code(
    "PYTHONPATH=. python -m tools.simulate_client_demo "
    "--server http://127.0.0.1:8080/events --channels 8 --realtime",
    language="bash",
)

st.markdown("Terminal 3:")
st.code("PYTHONPATH=. streamlit run dashboard/app.py", language="bash")

st.markdown("Terminal 4, optional dedicated spectrum page:")
st.code("PYTHONPATH=. streamlit run dashboard/station_spectrum_app.py --server.port 8502", language="bash")

st.subheader("What the operator should see")
st.markdown(
    """
- 0-8 sec: all simulated stations remain background.
- 8-18 sec: motorcycle-like audio may look noisy or suspect, but should not produce station ALERT.
- 18-34 sec: two stations see drone-like harmonic evidence and fusion rises.
- 34-48 sec: only one station remains affected, so fusion should drop from the multi-station case.
- 48-60 sec: all stations clear back toward background.
    """.strip()
)

st.subheader("What should not happen")
st.markdown(
    """
- Motorcycle should not produce ALERT.
- One station alone should not be treated as a public-warning candidate.
- HF/model output is advisory and must not alert without rotor-harmonic evidence.
    """.strip()
)
