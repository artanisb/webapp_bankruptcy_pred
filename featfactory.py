import numpy as np
import pandas as pd

class FeatureEngineer:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.generated_dfs = []

    # Erstellt alle Feature-Preset-Dateframes
    def build_dfs(self, df):
        df = df.sort_values(by=['year']).reset_index(drop=True)
        features_initial = [f'X{i}' for i in range(1, 19)]
        eps = self.eps

        def build_lags(period):
            lag_cols = {}
            for feat in features_initial:
                lag_feat = f'{feat}_lag{period}'
                lag_series = df[feat].shift(period)
                lag_series = lag_series.where(~((lag_series.isna()) | (lag_series == 0)), df[feat])
                lag_cols[lag_feat] = lag_series
            return pd.DataFrame(lag_cols)

        # Year-Shift um bis zu 3 Jahre
        lag1_df = build_lags(1)
        lag2_df = build_lags(2)
        lag3_df = build_lags(3)

        # Summe der letzten 3 Jahre
        sum3_df = df[features_initial].rolling(3, min_periods=1).sum()
        sum3_df.columns = [f'{col}_sum3' for col in sum3_df.columns]

        # Durschnitt aus den letzten 3 Jahren
        avg3_df = df[features_initial].rolling(3, min_periods=1).mean()
        avg3_df.columns = [f'{col}_avg3' for col in avg3_df.columns]

        # Flags
        flags_cols = {
            'NWC': (df['X14'] > df['X1']).astype(int),
            'NI_lag1': df['X6'].shift(1).fillna(df['X6']),
            'NI_lag2': df['X6'].shift(2).fillna(df['X6']),
            'ILTWO': ((df['X6'].shift(1) < 0) |
                      (df['X6'].shift(2) < 0)).astype(int),
            'NSL1': df['X9'].shift(1).fillna(df['X9']),
            'DNS': (df['X9'] < df['X9'].shift(1).fillna(df['X9'])).astype(int),
            'CFN': (df['X4'] < 0).astype(int)
        }
        flags_df = pd.DataFrame(flags_cols)

        # Ratios
        ratios_df = pd.DataFrame({
            'CUR_R': df['X1'] / (df['X14'] + eps),
            'QUI_R': (df['X1'] - df['X2']) / (df['X14'] + eps),
            'CAS_R': df['X3'] / (df['X14'] + eps),
            'DEEQ': df['X16'] / (df['X7'] + eps),
            'LEV_R': df['X10'] / (df['X7'] + eps),
            'INTC': df['X12'] / (df['X5'] + eps),
            'ROA': df['X6'] / (df['X10'] + eps),
            'ROE': df['X6'] / (df['X7'] + eps),
            'GPM': (df['X9'] - df['X11']) / (df['X9'] + eps),
            'OPM': df['X4'] / (df['X9'] + eps),
            'NPM': df['X6'] / (df['X9'] + eps),
            'AST': df['X9'] / (df['X10'] + eps),
            'INVT': df['X11'] / (df['X2'] + eps),
            'MABO': df['X8'] / (df['X18'] + eps),
            'DIVP': df['X13'] / (df['X6'] + eps)
        })

        # Ohlson Komponenten
        NITA = df['X6'] / (df['X10'] + eps)
        NITA_lag1 = NITA.shift(1).fillna(NITA)
        NITA_lag2 = NITA.shift(2).fillna(NITA)
        NITA_diff = NITA - NITA_lag1
        CHIN = NITA_diff / (NITA.abs() + NITA_lag1.abs() + eps)

        ocomp_df = pd.DataFrame({
            'TLTA': df['X17'] / (df['X10'] + eps),
            'log_TLTA': np.log(df['X17'] / (df['X10'] + eps) + eps),
            'WCTA': (df['X1'] - df['X14']) / (df['X10'] + eps),
            'CLCA': df['X14'] / (df['X1'] + eps),
            'OENEG': (df['X17'] > df['X10']).astype(int),
            'NITA': NITA,
            'FUTL': (df['X6'] + df['X3']) / (df['X17'] + eps),
            'NITA_lag1': NITA_lag1,
            'NITA_lag2': NITA_lag2,
            'INTWO': ((NITA_lag1 < 0) & (NITA_lag2 < 0)).astype(int),
            'NITA_diff': NITA_diff,
            'CHIN': CHIN,
        })

        # Ohlson O-Score
        ocomp_df['o_score'] = (
            -1.32 - 0.407 * ocomp_df['log_TLTA'] + 6.03 * ocomp_df['WCTA'] -
            1.43 * ocomp_df['CLCA'] + 0.0757 * ocomp_df['OENEG'] - 2.37 * ocomp_df['NITA'] -
            1.83 * ocomp_df['FUTL'] + 0.285 * ocomp_df['INTWO'] - 0.521 * ocomp_df['CHIN']
        )

        # Altman Komponenten
        zcomp_df = pd.DataFrame({
            'WC_TA': (df['X1'] - df['X14']) / (df['X10'] + eps),
            'RE_TA': df['X15'] / (df['X10'] + eps),
            'EBIT_TA': df['X12'] / (df['X10'] + eps),
            'MVE_TL': df['X8'] / (df['X17'] + eps),
            'S_TA': df['X9'] / (df['X10'] + eps),
        })

        # Altman Z-Score
        zcomp_df['z_score'] = (
            1.2 * zcomp_df['WC_TA'] +
            1.4 * zcomp_df['RE_TA'] +
            3.3 * zcomp_df['EBIT_TA'] +
            0.6 * zcomp_df['MVE_TL'] +
            1.0 * zcomp_df['S_TA']
        )

        self.generated_dfs = [df, lag1_df, lag2_df, lag3_df, sum3_df, avg3_df, flags_df, ratios_df, ocomp_df, zcomp_df]
        return self.generated_dfs

    # Verknüpft die Feature-Presets mit dem initialen Datensatz, basierend auf den Feature Flags
    def generate_features(self, include):
        dfs = self.generated_dfs
        to_concat = [dfs[0]]
        to_drop = set()

        if include.get('lag1'):
            to_concat.append(dfs[1])
        if include.get('lag2'):
            to_concat.append(dfs[2])
        if include.get('lag3'):
            to_concat.append(dfs[3])
        if include.get('sum3'):
            to_concat.append(dfs[4])
        if include.get('avg3'):
            to_concat.append(dfs[5])
        if include.get('flags'):
            to_concat.append(dfs[6])
        if include.get('ratios'):
            to_concat.append(dfs[7])
        if include.get('ocomp') or include.get('oscore'):
            to_concat.append(dfs[8])
        if include.get('zcomp') or include.get('zscore'):
            to_concat.append(dfs[9])

        if include.get('flags'):
            to_drop.update(['NI_lag1', 'NI_lag2', 'NSL1']) 
        if include.get('ratios') and include.get('ocomp'):
            to_drop.update(['NITA'])  
        if include.get('ratios') and include.get('zcomp'):
            to_drop.update(['S_TA'])
        if include.get('ocomp') and include.get('zcomp'):
            to_drop.update(['WC_TA'])  
        if not include.get('oscore') and include.get('ocomp'):
            to_drop.update(['NITA_lag1', 'NITA_lag2', 'NITA_diff', 'TLTA', 'o_score'])
        if include.get('oscore') and not include.get('ocomp'):
            to_drop.update(['NITA_lag1', 'NITA_lag2', 'NITA_diff', 'TLTA', 'log_TLTA', 'WCTA', 'CLCA', 'OENEG', 'NITA', 'FUTL', 'INTWO', 'CHIN'])
        if not include.get('zscore') and include.get('zcomp'):
            to_drop.update(['z_score'])
        if include.get('zscore') and not include.get('zcomp'):
            to_drop.update(['WC_TA', 'RE_TA', 'EBIT_TA', 'MVE_TL', 'S_TA'])

        to_drop.update(['X4', 'X13', 'X16'])
        df = pd.concat(to_concat, axis=1).drop(columns=list(to_drop))

        return df
    
    @staticmethod
    def build_features(data, features):
        fe = FeatureEngineer()
        fe.build_dfs(data)
        data = fe.generate_features(features)
        return data