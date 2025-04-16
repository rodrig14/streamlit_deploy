#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 00:31:05 2025

@author: rodrigue
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from fpdf import FPDF
import tempfile
import os
import requests
import xarray as xr
import pandas as pd
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO

# Configuration de l'application
st.set_page_config(
    page_title="Système d'Alerte aux Inondations",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement des styles CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")

# Constantes
DEFAULT_LOCATION = {"lat": 4.05, "lon": 9.7, "name": "Douala, Cameroun"}
THRESHOLDS = {
    "low": 50,
    "medium": 100,
    "high": 150
}

# Classes utilitaires
class DataProcessor:
    @staticmethod
    def process_netcdf(file):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name

            ds = xr.open_dataset(tmp_path)
            return ds, tmp_path
        except Exception as e:
            st.error(f"Erreur de traitement du fichier NetCDF: {str(e)}")
            return None, None

    @staticmethod
    def fetch_weather_data(lat, lon, api_key):
        try:
            url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Erreur API météo: {str(e)}")
            return None

class RiskCalculator:
    @staticmethod
    def normalize(val, vmin, vmax):
        return (val - vmin) / (vmax - vmin) if vmax != vmin else 0

    @staticmethod
    def calculate_iri(params, weights):
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        iri = (
            normalized_weights["intensity"] * params["intensity_norm"] +
            normalized_weights["duration"] * params["duration_norm"] +
            normalized_weights["accumulation"] * params["accumulation_norm"] +
            normalized_weights["humidity"] * params["humidity_norm"] +
            normalized_weights["slope"] * params["slope_norm"] +
            normalized_weights["land_use"] * params["land_use_norm"]
        )
        return iri

    @staticmethod
    def determine_risk_level(iri):
        if iri >= 0.7:
            return "high", "🔴 Risque Très Élevé"
        elif iri >= 0.4:
            return "medium", "🟠 Risque Élevé"
        elif iri >= 0.2:
            return "low", "🟡 Risque Modéré"
        else:
            return "none", "🟢 Risque Faible"

class MapVisualizer:
    @staticmethod
    def create_risk_map(lat, lon, risk_level, popup_content):
        color_map = {
            "high": "red",
            "medium": "orange",
            "low": "yellow",
            "none": "green"
        }
        
        m = folium.Map(location=[lat, lon], zoom_start=12)
        
        folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color=color_map[risk_level], icon="info-sign")
        ).add_to(m)
        
        folium.Circle(
            location=[lat, lon],
            radius=2000,
            color=color_map[risk_level],
            fill=True,
            fill_opacity=0.2
        ).add_to(m)
        
        folium.TileLayer('Stamen Terrain').add_to(m)
        folium.LayerControl().add_to(m)
        
        return m

class ReportGenerator:
    @staticmethod
    def create_pdf_report(data):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Header
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Rapport d'Analyse de Risque d'Inondation", ln=True, align='C')
        pdf.ln(10)
        
        # Date
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt=f"Date du rapport: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
        pdf.ln(5)
        
        # Location
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Localisation:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt=f"Latitude: {data['latitude']}, Longitude: {data['longitude']}", ln=True)
        pdf.ln(5)
        
        # Parameters
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Paramètres:", ln=True)
        pdf.set_font("Arial", size=10)
        
        params = [
            f"Source des données: {data['data_source']}",
            f"Intensité de pluie: {data['intensity']:.1f} mm/h",
            f"Durée: {data['duration']} heures",
            f"Cumul: {data['accumulation']:.1f} mm",
            f"Occupation du sol: {data['land_use']}",
            f"Humidité du sol: {data['humidity']}%",
            f"Pente: {data['slope']}%"
        ]
        
        for param in params:
            pdf.cell(200, 10, txt=param, ln=True)
        
        pdf.ln(5)
        
        # Results
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Résultats:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt=f"Indice IRI: {data['iri']:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"Niveau de risque: {data['risk_level']}", ln=True)
        pdf.ln(10)
        
        # Recommendations
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Recommandations:", ln=True)
        pdf.set_font("Arial", size=10)
        
        recommendations = {
            "high": "Prenez des mesures immédiates. Évacuation recommandée pour les zones basses.",
            "medium": "Soyez prêt à évacuer. Surveillez les alertes météo.",
            "low": "Restez informé. Vérifiez vos plans d'urgence.",
            "none": "Aucune action immédiate requise."
        }
        
        pdf.multi_cell(0, 10, txt=recommendations[data['risk_level_category']])
        
        return pdf

# Interface utilisateur
def main():
    st.title("🌊 Système d'Alerte Précoce aux Inondations")
    st.markdown("""
    Cet outil évalue le risque d'inondation en combinant des données météorologiques, 
    des caractéristiques du terrain et des facteurs environnementaux.
    """)
    
    # Initialisation de session
    if 'data' not in st.session_state:
        st.session_state.data = {
            "intensity": 0,
            "duration": 0,
            "accumulation": 0,
            "humidity": 50,
            "slope": 10,
            "land_use": "Urbain dense",
            "weights": {
                "intensity": 0.3,
                "duration": 0.2,
                "accumulation": 0.15,
                "humidity": 0.15,
                "slope": 0.1,
                "land_use": 0.1
            }
        }
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Sélection de la source de données
        data_source = st.radio(
            "Source des données",
            ["Saisie manuelle", "Fichier NetCDF", "API météo"],
            index=0
        )
        
        # Paramètres de localisation
        st.subheader("📍 Localisation")
        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=DEFAULT_LOCATION["lat"], format="%.4f")
        with col2:
            lon = st.number_input("Longitude", value=DEFAULT_LOCATION["lon"], format="%.4f")
        
        # Paramètres selon la source de données
        if data_source == "Saisie manuelle":
            st.subheader("🌦️ Données météorologiques")
            st.session_state.data["intensity"] = st.slider("Intensité de pluie (mm/h)", 0, 100, 30)
            st.session_state.data["duration"] = st.slider("Durée (heures)", 0, 72, 6)
            st.session_state.data["accumulation"] = st.slider("Cumul (mm)", 0, 500, 90)
        
        elif data_source == "Fichier NetCDF":
            st.subheader("📁 Fichier NetCDF")
            uploaded_file = st.file_uploader("Téléverser un fichier", type=["nc", "netcdf"])
            
            if uploaded_file:
                ds, tmp_path = DataProcessor.process_netcdf(uploaded_file)
                
                if ds:
                    # Sélection des variables
                    variables = list(ds.variables)
                    selected_var = st.selectbox("Variable de précipitation", variables)
                    
                    # Traitement des données
                    if selected_var:
                        try:
                            # Exemple simplifié - à adapter selon la structure de vos fichiers
                            precipitation = ds[selected_var]
                            st.session_state.data["accumulation"] = float(precipitation.mean().values)
                            st.session_state.data["duration"] = 24  # À adapter
                            st.session_state.data["intensity"] = st.session_state.data["accumulation"] / st.session_state.data["duration"]
                            
                            st.success("Données chargées avec succès!")
                        except Exception as e:
                            st.error(f"Erreur de traitement: {str(e)}")
                    
                    # Nettoyage
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        
        elif data_source == "API météo":
            st.subheader("🌤️ API Météo")
            api_key = st.text_input("Clé API OpenWeatherMap", type="password")
            
            if api_key:
                weather_data = DataProcessor.fetch_weather_data(lat, lon, api_key)
                
                if weather_data:
                    # Analyse des prévisions
                    hourly_rain = []
                    for forecast in weather_data["list"][:8]:  # Prochaines 24h
                        rain = forecast.get("rain", {}).get("3h", 0)
                        hourly_rain.append(rain)
                    
                    st.session_state.data["accumulation"] = sum(hourly_rain)
                    st.session_state.data["duration"] = len(hourly_rain) * 3
                    st.session_state.data["intensity"] = max(hourly_rain) if hourly_rain else 0
                    
                    st.success(f"Données météo actualisées (prochaines 24h)")
        
        # Paramètres du terrain
        st.subheader("🏞️ Caractéristiques du terrain")
        st.session_state.data["land_use"] = st.selectbox(
            "Occupation du sol",
            ["Urbain dense", "Urbain dispersé", "Forêt", "Savane", "Zone agricole", "Zone humide"]
        )
        st.session_state.data["humidity"] = st.slider("Humidité du sol (%)", 0, 100, 50)
        st.session_state.data["slope"] = st.slider("Pente du terrain (%)", 0, 100, 10)
        
        # Pondérations
        st.subheader("⚖️ Pondérations")
        st.session_state.data["weights"]["intensity"] = st.slider("Intensité", 0.0, 1.0, 0.3)
        st.session_state.data["weights"]["duration"] = st.slider("Durée", 0.0, 1.0, 0.3)
        st.session_state.data["weights"]["accumulation"] = st.slider("Cumul", 0.0, 1.0, 0.15)
        st.session_state.data["weights"]["humidity"] = st.slider("Humidité", 0.0, 1.0, 0.15)
        st.session_state.data["weights"]["slope"] = st.slider("Pente", 0.0, 1.0, 0.1)
        st.session_state.data["weights"]["land_use"] = st.slider("Occupation sol", 0.0, 1.0, 0.1)
    
    # Calcul des paramètres de risque
    land_use_factors = {
        "Urbain dense": 1.0,
        "Urbain dispersé": 0.8,
        "Zone agricole": 0.6,
        "Savane": 0.4,
        "Forêt": 0.2,
        "Zone humide": 0.9
    }
    
    params = {
        "intensity_norm": RiskCalculator.normalize(st.session_state.data["intensity"], 0, 100),
        "duration_norm": RiskCalculator.normalize(st.session_state.data["duration"], 0, 72),
        "accumulation_norm": RiskCalculator.normalize(st.session_state.data["accumulation"], 0, 500),
        "humidity_norm": RiskCalculator.normalize(st.session_state.data["humidity"], 0, 100),
        "slope_norm": RiskCalculator.normalize(st.session_state.data["slope"], 0, 100),
        "land_use_norm": land_use_factors[st.session_state.data["land_use"]]
    }
    
    # Calcul de l'IRI
    iri = RiskCalculator.calculate_iri(params, st.session_state.data["weights"])
    risk_level, risk_label = RiskCalculator.determine_risk_level(iri)
    
    # Affichage des résultats
    st.header("📊 Résultats d'analyse")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Indice IRI", f"{iri:.2f}")
    with col2:
        st.metric("Niveau de risque", risk_label)
    with col3:
        st.metric("Cumul de pluie", f"{st.session_state.data['accumulation']:.1f} mm")
    
    # Carte interactive
    st.subheader("🗺️ Cartographie du risque")
    popup_content = f"""
    <div style="width: 250px">
        <h4>Analyse de risque d'inondation</h4>
        <p><b>Localisation:</b> {lat:.4f}, {lon:.4f}</p>
        <p><b>Indice IRI:</b> {iri:.2f}</p>
        <p><b>Niveau:</b> {risk_label}</p>
        <p><b>Dernière mise à jour:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    """
    risk_map = MapVisualizer.create_risk_map(lat, lon, risk_level, popup_content)
    folium_static(risk_map, width=900, height=500)
    
    # Graphique des composantes
    st.subheader("📈 Composition du risque")
    components = pd.DataFrame({
        "Composante": ["Intensité", "Durée", "Cumul", "Humidité", "Pente", "Sol"],
        "Valeur": [
            params["intensity_norm"],
            params["duration_norm"],
            params["accumulation_norm"],
            params["humidity_norm"],
            params["slope_norm"],
            params["land_use_norm"]
        ]
    })
    fig = px.bar(components, x="Composante", y="Valeur", color="Composante")
    st.plotly_chart(fig, use_container_width=True)
    
    # Génération du rapport
    st.subheader("📝 Rapport d'analyse")
    if st.button("Générer le rapport complet"):
        report_data = {
            "latitude": lat,
            "longitude": lon,
            "data_source": data_source,
            "intensity": st.session_state.data["intensity"],
            "duration": st.session_state.data["duration"],
            "accumulation": st.session_state.data["accumulation"],
            "humidity": st.session_state.data["humidity"],
            "slope": st.session_state.data["slope"],
            "land_use": st.session_state.data["land_use"],
            "iri": iri,
            "risk_level": risk_label,
            "risk_level_category": risk_level
        }
        
        pdf = ReportGenerator.create_pdf_report(report_data)
        
        # Sauvegarde temporaire du PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            pdf.output(tmp_file.name)
            with open(tmp_file.name, "rb") as f:
                pdf_bytes = f.read()
        
        # Création du lien de téléchargement
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="rapport_inondation.pdf">Télécharger le rapport PDF</a>'
        st.markdown(href, unsafe_allow_html=True)
        os.unlink(tmp_file.name)

if __name__ == "_main_":
   main()