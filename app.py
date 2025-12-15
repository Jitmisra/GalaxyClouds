import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from galaxyclouds.io import generate_synthetic_galaxies
from galaxyclouds.observables import compute_all_observables

st.set_page_config(page_title="GalaxyClouds Dashboard", layout="wide")

st.title("🌌 GalaxyClouds: Point Cloud Morphology")

@st.cache_data
def load_data():
    X, y = generate_synthetic_galaxies(n_per_class=200, seed=42)
    mask = X[:, :, 0] > 0
    df = compute_all_observables(X, mask)
    df['class'] = [ {0: 'Elliptical', 1: 'Spiral', 2: 'Irregular'}[l] for l in y ]
    return X, y, mask, df

X, y, mask, df = load_data()

st.sidebar.header("Controls")
view_mode = st.sidebar.radio("View Mode", ["Observable Distributions", "Point Cloud Viewer"])

if view_mode == "Observable Distributions":
    st.subheader("Observable Distributions by Morphology")
    
    obs = st.selectbox("Select Observable", df.columns[:-1])
    
    fig = px.histogram(df, x=obs, color='class', barmode='overlay', 
                      marginal='box', hover_data=df.columns,
                      color_discrete_sequence=['#4B0082', '#FF8C00', '#2E8B57'])
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Physical Intuition**: Ellipticals tend to have lower multiplicity and higher central concentration, whereas spirals and irregulars exhibit more extended structures and higher multiplicity due to distributed star formation.
    """)

else:
    st.subheader("Interactive Galaxy Point Cloud")
    
    idx = st.slider("Galaxy Index", 0, len(y)-1, 0)
    
    n_stars = int(np.sum(mask[idx]))
    flux = X[idx, :n_stars, 0]
    ra = X[idx, :n_stars, 1]
    dec = X[idx, :n_stars, 2]
    
    galaxy_class = df.iloc[idx]['class']
    
    fig = go.Figure(data=[go.Scatter(
        x=ra, y=dec, mode='markers',
        marker=dict(
            size=flux / flux.max() * 20 + 2,
            color=flux,
            colorscale='Viridis',
            showscale=True
        )
    )])
    fig.update_layout(
        title=f"Galaxy #{idx} ({galaxy_class}) - {n_stars} stars",
        xaxis_title="RA Offset (deg)",
        yaxis_title="Dec Offset (deg)",
        template='plotly_dark',
        width=800, height=600
    )
    # maintain aspect ratio
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.write(df.iloc[idx:idx+1])
