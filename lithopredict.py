import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

lithology_keys = {0: 'Sandstone', 1: 'Sandstone/Shale', 2: 'Shale', 3: 'Marl', 4: 'Dolomite', 5: 'Limestone', 6: 'Chalk', 7: 'Halite', 8: 'Anhydrite', 9: 'Tuff', 10: 'Coal', 11: 'Basement'}

# Instructions for the user
st.markdown("""
    ### Instructions
    - Please upload a CSV file containing well logs.
    - The following logs must be present: DEPTH_MD, DRHO, NPHI, DTC, GR, PEF, RDEP, RHOB, CALI, RMED.
    - Fill any missing values with zeros before uploading.
    - Use the slider to select the depth range for visualization.
""")

def predict_lithology(file_path):
    model = load_model('model.h5')
    scaler = load('my_scaler4.pkl')
    df = pd.read_csv(file_path, delimiter=',')
    df['NPHI_to_GR'] = df['NPHI'] / df['GR']
    selected_features = ['DEPTH_MD','DRHO', 'NPHI', 'DTC', 'GR','PEF', 'RDEP', 'RHOB', 'CALI', 'RMED', 'NPHI_to_GR']
    X = df[selected_features].copy()
    X.fillna(0, inplace=True)
    X_normalized = scaler.transform(X)
    X_normalized_reshaped = X_normalized.reshape(X_normalized.shape[0], X_normalized.shape[1], 1)
    predictions = model.predict(X_normalized_reshaped)
    predictions = np.argmax(predictions, axis=1)
    predictions = [lithology_keys[pred] for pred in predictions]
    df['Predicted_Lithology'] = predictions
    return df

st.title('Lithology Prediction from Well Logs')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.write("Predicting...")
    df = predict_lithology(uploaded_file)
    st.write("Prediction complete!")
    st.write("Predicted Lithology Log:")
    st.dataframe(df[['DEPTH_MD', 'Predicted_Lithology']])
    csv = df.to_csv(index=False)
    st.download_button('Download Predicted Lithology', csv, file_name='predicted_lithology.csv')

    min_depth = df['DEPTH_MD'].min()
    max_depth = df['DEPTH_MD'].max()
    depth_range = st.slider('Select Depth Range:', min_value=float(min_depth), max_value=float(max_depth), value=(float(min_depth), float(max_depth)))
    df_filtered = df[(df['DEPTH_MD'] >= depth_range[0]) & (df['DEPTH_MD'] <= depth_range[1])]

    def plot_lithology_log(df_to_plot, ax):
        depth_values = df_to_plot['DEPTH_MD'].values
        lithology_mapping = {lithology: i for i, lithology in enumerate(lithology_keys.values())}
        lithology_classes = [lithology_mapping[lithology] for lithology in df_to_plot['Predicted_Lithology']]
        cmap = plt.cm.get_cmap('Paired', len(lithology_keys))
        norm = mcolors.BoundaryNorm(np.arange(-0.5, len(lithology_keys), 1), cmap.N)
        im = ax.imshow(np.array([lithology_classes]).T, aspect='auto', cmap=cmap, norm=norm)
        ax.set_yticks(np.arange(0, len(depth_values), 500))
        ax.set_yticklabels([f"{d:.2f}" for d in depth_values[::500]])
        ax.set_xticks([])
        return im

    # Create two columns for side-by-side plots
    col1, col2 = st.columns(2)

    # Full plot with highlighted region
    fig_full, ax_full = plt.subplots(figsize=(2, 10))
    plot_lithology_log(df, ax_full)
    ax_full.axhspan(depth_range[0], depth_range[1], facecolor='red', alpha=0.3)  # Highlight selected range
    plt.title('Full Lithology Log')
    col1.pyplot(fig_full)  # Display in the first column

    # Zoomed-in plot
    fig_zoom, ax_zoom = plt.subplots(figsize=(2, 10))
    im_zoom = plot_lithology_log(df_filtered, ax_zoom)
    cbar = fig_zoom.colorbar(im_zoom, ax=ax_zoom, ticks=np.arange(len(lithology_keys)))
    cbar.ax.set_yticklabels(lithology_keys.values())
    plt.title('Zoomed-in Lithology Log')
    col2.pyplot(fig_zoom)  # Display in the second column
