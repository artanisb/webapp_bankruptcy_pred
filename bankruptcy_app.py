import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import application_pipeline as ap

max_years = 6 

# Page configuration
st.set_page_config(
    page_title="Insolvenz-Vorhersage",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("🏢 Insolvenz-Vorhersage Tool")
st.markdown("---")
st.write("Geben Sie die Finanzkennzahlen des Unternehmens für mehrere Jahre ein, um das Insolvenzrisiko zu bewerten.")

# Add CSV upload option
st.markdown("### 📁 Oder laden Sie eine CSV-Datei hoch")
with st.expander("ℹ️ CSV-Datei Format anzeigen", expanded=False):
    st.markdown("""
    **CSV-Datei Format:**
    - Die CSV-Datei muss eine Spalte 'year' und die Kennzahlen x1-x18 enthalten
    - Die Jahre sollten in aufsteigender oder absteigender Reihenfolge sein
    - Maximal 6 Jahre werden berücksichtigt
    - Alle Werte sind in Tsd. Euro anzugeben (1 = 1.000€)
    - Beispiel:
    ```
    year,x1,x2,x3,x5,x6,x7,x8,x9,x10,x11,x12,x14,x15,x17,x18
    2023,1000,500,100,200,300,150,800,2000,3000,1000,400,250,200,1500,800
    2022,900,450,90,180,270,135,720,1800,2700,900,360,225,180,1350,720
    2021,800,400,80,160,240,120,640,1600,2400,800,320,200,160,1200,640
    ```
    """)

uploaded_file = st.file_uploader("Wählen Sie eine CSV-Datei mit den Finanzkennzahlen", type=['csv'], help="Die CSV-Datei muss die Spalten 'year' und X1-X18 enthalten. Die Jahre sollten chronologisch geordnet sein.")

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        df.columns = [col.strip().upper() if col.strip().lower() != 'year' else 'year' for col in df.columns]
        
        # Check if all required columns are present
        required_columns = ['year', 'X1', 'X2', 'X3', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 
                          'X11', 'X12', 'X14', 'X15', 'X17', 'X18']
        
        if all(col in df.columns for col in required_columns):
            # Sort by year in descending order (newest first)
            df = df.sort_values('year', ascending=False)
            
            # Validate years
            current_year = datetime.now().year
            if df['year'].max() > current_year:
                st.warning("⚠️ Die Daten enthalten Jahre in der Zukunft. Diese werden ignoriert.")
                df = df[df['year'] <= current_year]
            
            if len(df) > max_years:
                st.warning(f"⚠️ Die Daten enthalten mehr als {max_years} Jahre. Nur die neuesten {max_years} Jahre werden verwendet.")
                df = df.head(max_years)
            
            st.success("✅ CSV-Datei erfolgreich geladen!")
            
            # Store the data in session state
            st.session_state.yearly_data = {}  # Reset existing data
            for _, row in df.iterrows():
                year = int(row['year'])
                st.session_state.yearly_data[year] = {
                    'X1': row['X1'], 'X2': row['X2'], 'X3': row['X3'], 
                    'X5': row['X5'], 'X6': row['X6'], 'X7': row['X7'], 
                    'X8': row['X8'], 'X9': row['X9'], 'X10': row['X10'], 
                    'X11': row['X11'], 'X12': row['X12'], 'X14': row['X14'], 
                    'X15': row['X15'], 'X17': row['X17'], 'X18': row['X18']
                }
            
            # Update number of years based on CSV data
            num_years = len(df)
            years = sorted(st.session_state.yearly_data.keys(), reverse=True)
            
            # Show preview of loaded data
            st.markdown("### 📊 Vorschau der geladenen Daten")
            preview_df = df.copy()
            preview_df = preview_df.sort_values('year', ascending=False)
            st.dataframe(preview_df, use_container_width=True)
            
        else:
            missing_columns = [col for col in required_columns if col not in df.columns]
            st.error(f"❌ Fehlende Spalten in der CSV-Datei: {', '.join(missing_columns)}")
            st.info("Die CSV-Datei muss folgende Spalten enthalten: " + ", ".join(required_columns))
            st.info("Beispiel-Format:\n```\nyear,X1,X2,X3,X5,X6,X7,X8,X9,X10,X11,X12,X14,X15,X17,X18\n2023,1000000,500000,100000,200000,300000,150000,800000,2000000,3000000,1000000,400000,250000,200000,1500000,800000\n2022,900000,450000,90000,180000,270000,135000,720000,1800000,2700000,900000,360000,225000,180000,1350000,720000\n```")
            
    except Exception as e:
        st.error(f"❌ Fehler beim Lesen der CSV-Datei: {str(e)}")
        st.info("Bitte stellen Sie sicher, dass die CSV-Datei das korrekte Format hat.")

st.markdown("---")
st.write("### ✏️ Oder geben Sie die Daten manuell ein:")

# Initialize session state for storing yearly data
if 'yearly_data' not in st.session_state:
    st.session_state.yearly_data = {}

# Only show manual input if no CSV is uploaded
if uploaded_file is None:
    # Year selection
    current_year = datetime.now().year
    max_years = 15  # Maximum number of years to look back

    # Number of years selection
    num_years = st.number_input(
        "Anzahl der Jahre",
        min_value=1,
        max_value=max_years,
        value=6,
        help="Wählen Sie, wie viele Jahre Sie analysieren möchten (aktuelles Jahr und Jahre davor)"
    )

    # Calculate years to display (current year and years before)
    years = list(range(current_year - num_years + 1, current_year + 1))

    # Create tabs for each year
    tabs = st.tabs([str(year) for year in years])

    # Function to create input fields for a specific year
    def create_input_fields(year):
        # Umsatz & Profitabilität
        st.subheader("📈 Umsatz & Profitabilität")
        col1, col2 = st.columns(2)
        with col1:
            X9 = st.number_input(
                "X9: Nettoumsatz (in Tsd. €)", 
                format="%.2f",
                help="Verkauf abzüglich Retouren, Zuschüssen und Rabatte (1 = 1.000€)",
                key=f"X9_{year}"
            )
            X12 = st.number_input(
                "X12: EBIT (in Tsd. €)", 
                format="%.2f",
                help="Gewinn vor Zinsen und Steuern (1 = 1.000€)",
                key=f"X12_{year}"
            )
        with col2:
            X6 = st.number_input(
                "X6: Gewinn nach Kosten (in Tsd. €)", 
                format="%.2f",
                help="Umsatz abzüglich aller Kosten (1 = 1.000€)",
                key=f"X6_{year}"
            )
            X15 = st.number_input(
                "X15: Bilanzgewinn (in Tsd. €)", 
                format="%.2f",
                help="Gewinn nach allen Kosten, Steuern und Auszahlungen (1 = 1.000€)",
                key=f"X15_{year}"
            )

        # Kosten & Aufwendungen
        st.subheader("💸 Kosten & Aufwendungen")
        col3, col4 = st.columns(2)
        with col3:
            X2 = st.number_input(
                "X2: Verkaufskosten (in Tsd. €)", 
                format="%.2f",
                help="Kosten aus dem Verkauf (1 = 1.000€)",
                key=f"X2_{year}"
            )
            X18 = st.number_input(
                "X18: Betriebskosten (in Tsd. €)", 
                format="%.2f",
                help="Personalkosten, Miete, Energie, Material etc. (1 = 1.000€)",
                key=f"X18_{year}"
            )
        with col4:
            X3 = st.number_input(
                "X3: Abschreibungen (in Tsd. €)", 
                format="%.2f",
                help="Abschreibung und Wertverlust (1 = 1.000€)",
                key=f"X3_{year}"
            )

        # Vermögen & Assets
        st.subheader("🏦 Vermögen & Assets")
        col5, col6 = st.columns(2)
        with col5:
            X1 = st.number_input(
                "X1: Alle Vermögenswerte (in Tsd. €)", 
                format="%.2f",
                help="Vermögenswerte für wirtschaftliche Transaktionen im nächsten Jahr (1 = 1.000€)",
                key=f"X1_{year}"
            )
            X5 = st.number_input(
                "X5: Inventar (in Tsd. €)", 
                format="%.2f",
                help="Güter und Rohstoffe für Produktion/Verkauf (1 = 1.000€)",
                key=f"X5_{year}"
            )
        with col6:
            X8 = st.number_input(
                "X8: Marktpreis Assets (in Tsd. €)", 
                format="%.2f",
                help="Marktpreis von Aktien, Anleihen etc. (1 = 1.000€)",
                key=f"X8_{year}"
            )
            X10 = st.number_input(
                "X10: Gesamtvermögen (in Tsd. €)", 
                format="%.2f",
                help="Vermögenswerte und Wertgegenstände (1 = 1.000€)",
                key=f"X10_{year}"
            )

        # Schulden & Verbindlichkeiten
        st.subheader("🔴 Schulden & Verbindlichkeiten")
        col7, col8 = st.columns(2)
        with col7:
            X7 = st.number_input(
                "X7: Forderungen an Dritte (in Tsd. €)", 
                format="%.2f",
                help="Noch nicht beglichene Forderungen (1 = 1.000€)",
                key=f"X7_{year}"
            )
            X11 = st.number_input(
                "X11: Langfristige Schulden (in Tsd. €)", 
                format="%.2f",
                help="Schulden, die dieses Jahr nicht fällig sind (1 = 1.000€)",
                key=f"X11_{year}"
            )
        with col8:
            X14 = st.number_input(
                "X14: Kurzfristige Verbindlichkeiten (in Tsd. €)", 
                format="%.2f",
                help="Lieferungen, Steuern, Gehälter etc. (1 = 1.000€)",
                key=f"X14_{year}"
            )
            X17 = st.number_input(
                "X17: Gesamte Verbindlichkeiten (in Tsd. €)", 
                format="%.2f",
                help="Alle Schulden und Rechnungen an Dritte (1 = 1.000€)",
                key=f"X17_{year}"
            )
        # Store the data in session state
        st.session_state.yearly_data[year] = {
            'X1': X1, 'X2': X2, 'X3': X3, 'X5': X5, 'X6': X6, 'X7': X7, 'X8': X8,
            'X9': X9, 'X10': X10, 'X11': X11, 'X12': X12, 'X14': X14, 'X15': X15,
            'X17': X17, 'X18': X18
        }

    # Create input fields for each year in tabs
    for i, year in enumerate(years):
        with tabs[i]:
            create_input_fields(year)
else:
    # If CSV is uploaded, show a message that manual input is hidden
    st.info("📁 CSV-Datei geladen - Manuelle Eingabe ausgeblendet. Entfernen Sie die CSV-Datei, um manuelle Eingabe zu aktivieren.")
    
    # Pre-fill the input fields with CSV data if available
    if 'yearly_data' in st.session_state and st.session_state.yearly_data:
        st.markdown("### 📊 Daten aus CSV-Datei:")
        
        # Get the years from CSV data
        csv_years = sorted(st.session_state.yearly_data.keys(), reverse=True)
        
        # Create tabs for each year from CSV
        csv_tabs = st.tabs([str(year) for year in csv_years])
        
        # Function to create read-only input fields for CSV data
        def create_csv_input_fields(year):
            data = st.session_state.yearly_data[year]
            # Umsatz & Profitabilität
            st.subheader("📈 Umsatz & Profitabilität")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("X9: Nettoumsatz (in Tsd. €)", f"{data['X9']:,.2f}")
                st.metric("X12: EBIT (in Tsd. €)", f"{data['X12']:,.2f}")
            with col2:
                st.metric("X6: Gewinn nach Kosten (in Tsd. €)", f"{data['X6']:,.2f}")
                st.metric("X15: Bilanzgewinn (in Tsd. €)", f"{data['X15']:,.2f}")
            # Kosten & Aufwendungen
            st.subheader("💸 Kosten & Aufwendungen")
            col3, col4 = st.columns(2)
            with col3:
                st.metric("X2: Verkaufskosten (in Tsd. €)", f"{data['X2']:,.2f}")
                st.metric("X18: Betriebskosten (in Tsd. €)", f"{data['X18']:,.2f}")
            with col4:
                st.metric("X3: Abschreibungen (in Tsd. €)", f"{data['X3']:,.2f}")
            # Vermögen & Assets
            st.subheader("🏦 Vermögen & Assets")
            col5, col6 = st.columns(2)
            with col5:
                st.metric("X1: Alle Vermögenswerte (in Tsd. €)", f"{data['X1']:,.2f}")
                st.metric("X5: Inventar (in Tsd. €)", f"{data['X5']:,.2f}")
            with col6:
                st.metric("X8: Marktpreis Assets (in Tsd. €)", f"{data['X8']:,.2f}")
                st.metric("X10: Gesamtvermögen (in Tsd. €)", f"{data['X10']:,.2f}")
            # Schulden & Verbindlichkeiten
            st.subheader("🔴 Schulden & Verbindlichkeiten")
            col7, col8 = st.columns(2)
            with col7:
                st.metric("X7: Forderungen an Dritte (in Tsd. €)", f"{data['X7']:,.2f}")
                st.metric("X11: Langfristige Schulden (in Tsd. €)", f"{data['X11']:,.2f}")
            with col8:
                st.metric("X14: Kurzfristige Verbindlichkeiten (in Tsd. €)", f"{data['X14']:,.2f}")
                st.metric("X17: Gesamte Verbindlichkeiten (in Tsd. €)", f"{data['X17']:,.2f}")
        
        # Create read-only display for each year from CSV
        for i, year in enumerate(csv_years):
            with csv_tabs[i]:
                create_csv_input_fields(year)

st.markdown("---")

# Prediction Button
if st.button("🔍 Insolvenzrisiko berechnen", type="primary"):
    # Check if we have data for all years
    if len(st.session_state.yearly_data) < len(years):
        st.error("❌ Bitte füllen Sie die Daten für alle Jahre aus.")
    else:
        # Prepare historical data for prediction
        historical_data = []
        for year in years:
            data = st.session_state.yearly_data[year]
            features = [
                data['X1'], data['X2'], data['X3'], data['X5'], data['X6'],
                data['X7'], data['X8'], data['X9'], data['X10'], data['X11'],
                data['X12'], data['X14'], data['X15'], data['X17'], data['X18']
            ]
            historical_data.append(features)

        latest_data = st.session_state.yearly_data[max(years)]
        debt_ratio = latest_data['X17'] / max(latest_data['X10'], 1)
        profit_margin = latest_data['X15'] / max(latest_data['X9'], 1) if latest_data['X9'] > 0 else -1
        risk_score = min(1.0, max(0.0, debt_ratio * 0.7 + (1 - profit_margin) * 0.3))
        
        # Use the ML model for prediction
        try:
            # Convert manual data to DataFrame format for the model
            data_list = []
            for year in years:
                data = st.session_state.yearly_data[year]
                row = {'year': year}
                row.update(data)
                data_list.append(row)
            
            df_model = pd.DataFrame(data_list)
            df_model = df_model.sort_values('year')
            required_columns = ['year', 'X1', 'X2', 'X3', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 
                              'X11', 'X12', 'X14', 'X15', 'X17', 'X18']
            df_model = df_model[required_columns]
            
            # Use GRU config for prediction
            hp = ap.hp_gru
            risk_pred = ap.compute_score(hp, df_model)
            print(risk_pred)
            
            # Extract the risk score
            if isinstance(risk_pred, (list, np.ndarray)):
                risk_score = float(risk_pred[0])
            else:
                risk_score = float(risk_pred)
            print(risk_score)
                
            st.success("✅ ML-Modell-Vorhersage erfolgreich berechnet!")
            
        except Exception as e:
            st.error(f"❌ Fehler bei der ML-Modell-Vorhersage: {type(e).__name__} - {e}")
            # Fallback to demo calculation if ML model fails
            st.warning("⚠️ Fallback auf Demo-Berechnung verwendet.")
        
        # Display results
        st.subheader("📊 Vorhersage für das Jahr " + str(current_year + 1))
        
        # Main binary decision with enhanced visibility
        insolvency_prediction = "Insolvenz" if risk_score > 0.5 else "keine Insolvenz"
        prediction_color = "red" if insolvency_prediction == "Insolvenz" else "green"
        
        st.markdown(f"""
        <div style='display: flex; justify-content: center; width: 100%;'>
            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin: 20px 0; width: 50%;'>
                <h2 style='color: {prediction_color}; font-size: 36px; margin-bottom: 10px;'>
                    {insolvency_prediction}
                </h2>
                <p style='font-size: 18px; color: #665;'>
                    Insolvenzvorhersage für das nächste Jahr
                </p>
                <p style='font-size: 16px; color: #665;'>
                    Wahrscheinlichkeit: {risk_score:.1%}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            debt_delta = "Kritisch" if debt_ratio > 0.8 else "Normal"
            debt_delta_value = -1 if debt_delta == "Kritisch" else 1
            st.metric(
                "Verschuldungsgrad", 
                f"{debt_ratio:.1%}",
                delta=debt_delta,
                delta_color="inverse" if debt_delta == "Kritisch" else "normal"
            )
        
        with col_result2:
            profit_delta = "Gut" if profit_margin > 0.3 else "Schlecht"
            profit_delta_value = 1 if profit_delta == "Gut" else -1
            st.metric(
                "Gewinnmarge", 
                f"{profit_margin:.1%}" if profit_margin >= 0 else "Negativ",
                delta=profit_delta,
                delta_color="normal" if profit_delta == "Gut" else "inverse"
            )
        
        with col_result3:
            st.metric(
                "Risikowahrscheinlichkeit",
                f"{risk_score:.1%}",
                delta="Gesamtrisiko"
            )
        
        # Risk categories
        st.subheader("🔍 Detaillierte Risikokategorisierung")
        
        if risk_score < 0.3:
            st.success("✅ Normales Monitoring ausreichend - Unternehmen erscheint finanziell stabil")
        elif risk_score < 0.7:
            st.warning("⚠️ Verstärkte Überwachung empfohlen - Regelmäßige Kontrolle der Finanzkennzahlen")
        else:
            st.error("🚨 Hohes Insolvenzrisiko - Sofortige Überprüfung und Maßnahmen erforderlich")
        
        # Additional ratios
        with st.expander("📈 Detaillierte Kennzahlen"):
            st.write("**Finanzielle Kennzahlen:**")
            st.write(f"- Eigenkapitalquote: {((latest_data['X10'] - latest_data['X17']) / max(latest_data['X10'], 1)):.1%}")
            st.write(f"- Liquidität (vereinfacht): {(latest_data['X1'] / max(latest_data['X14'], 1)):.2f}")
            st.write(f"- Umsatzrentabilität: {(latest_data['X6'] / max(latest_data['X9'], 1)):.1%}" if latest_data['X9'] > 0 else "- Umsatzrentabilität: N/A")

# Sidebar with additional information
st.sidebar.header("ℹ️ Informationen")
st.sidebar.write("""
**Modell-Features:**
- X1: Vermögenswerte
- X2: Verkaufskosten  
- X3: Abschreibungen
- X5: Inventar
- X6: Gewinn nach Kosten
- X7: Forderungen
- X8: Marktpreis Assets
- X9: Nettoumsatz
- X10: Gesamtvermögen
- X11: Langfristige Schulden
- X12: EBIT
- X14: Kurzf. Verbindlichkeiten
- X15: Bilanzgewinn
- X17: Gesamte Verbindlichkeiten
- X18: Betriebskosten
""")

st.sidebar.info("💡 Tipp: Alle Werte sind in Tsd. Euro anzugeben (1 = 1.000€).")

# Footer
st.markdown("---")
st.markdown("*Entwickelt für die Insolvenz-Risikoanalyse*")