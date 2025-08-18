# KUGELSICHERER FIX - Ersetze den gesamten Excel Export Block (ab Zeile ~320) mit:

        # Schritt 8: Excel Export
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_filename = f"enhanced_divergence_analysis_{timestamp}.xlsx"
        
        logger.info(f"📋 Exportiere Ergebnisse: {excel_filename}")
        
        try:
            # AGGRESSIVE Timezone-Entfernung für ALLE Spalten
            df_export = df.copy()
            
            # Methode 1: Gehe durch ALLE Spalten und repariere sie
            for col in df_export.columns:
                try:
                    # Prüfe ob es sich um eine datetime-Spalte handelt
                    if (str(df_export[col].dtype).startswith('datetime64') or 
                        'date' in col.lower() or 
                        'time' in col.lower()):
                        
                        # Konvertiere zu timezone-naive datetime
                        df_export[col] = pd.to_datetime(df_export[col], errors='coerce').dt.tz_localize(None)
                        logger.debug(f"✅ Fixed column: {col}")
                        
                except Exception as e:
                    logger.debug(f"⚠️ Could not fix column {col}: {e}")
                    # Falls immer noch problematisch, konvertiere zu String
                    try:
                        if 'date' in col.lower():
                            df_export[col] = df_export[col].astype(str)
                    except:
                        pass
            
            # Methode 2: Spezifische bekannte Problemspalten
            problem_columns = [
                'date', 'Date', 'DATE',
                'CBullD_Lower_Low_date_gen', 'CBullD_Higher_Low_date_gen',
                'CBullD_Lower_Low_date_neg_MACD', 'CBullD_Higher_Low_date_neg_MACD',
                'HBullD_Lower_Low_date', 'HBullD_Higher_Low_date',
                'BearD_Higher_High_date', 'BearD_Lower_High_date'
            ]
            
            for col in problem_columns:
                if col in df_export.columns:
                    try:
                        df_export[col] = pd.to_datetime(df_export[col], errors='coerce').dt.tz_localize(None)
                    except:
                        # Als letzter Ausweg: als String speichern
                        df_export[col] = df_export[col].astype(str)
            
            # Methode 3: Falls immer noch Probleme, alle object-Spalten prüfen
            for col in df_export.select_dtypes(include=['object']).columns:
                if len(df_export) > 0:
                    try:
                        sample = df_export[col].dropna().iloc[0] if not df_export[col].dropna().empty else None
                        if sample and hasattr(sample, 'tzinfo'):
                            df_export[col] = pd.to_datetime(df_export[col], errors='coerce').dt.tz_localize(None)
                    except:
                        pass
            
            # JETZT sollte der Excel Export funktionieren
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                # Komplette Daten
                df_export.to_excel(writer, sheet_name='Complete_Data', index=False)
                
                # Zusammenfassung
                summary_data = []
                summary_data.append(['DIVERGENZ-STATISTIKEN', ''])
                for div_type, count in divergence_stats.items():
                    summary_data.append([div_type, count])
                summary_data.append(['TOTAL', total_divergences])
                summary_data.append(['', ''])
                summary_data.append(['PARAMETER', ''])
                summary_data.append(['Window Size', 5])
                summary_data.append(['Candle Tolerance', 0.1])
                summary_data.append(['MACD Tolerance', 3.25])
                
                summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Alle Divergenzen - AUCH HIER timezone-fix!
                all_divergences = []
                for div_col in ['CBullD_gen', 'CBullD_neg_MACD', 'HBullD', 'BearD']:
                    if div_col in df.columns:
                        div_data = df[df[div_col] == 1].copy()
                        if not div_data.empty:
                            div_data['Divergence_Type'] = div_col
                            # Timezone auch hier entfernen!
                            if 'date' in div_data.columns:
                                div_data['date'] = pd.to_datetime(div_data['date']).dt.tz_localize(None)
                            all_divergences.append(div_data[['date', 'close', 'RSI', 'macd_histogram', 'Divergence_Type']])
                
                if all_divergences:
                    combined_div = pd.concat(all_divergences, ignore_index=True)
                    combined_div = combined_div.sort_values('date').reset_index(drop=True)
                    combined_div.to_excel(writer, sheet_name='All_Divergences', index=False)
            
            logger.info(f"✅ Excel erfolgreich exportiert: {excel_filename}")
            
        except Exception as e:
            logger.error(f"❌ Excel Export fehlgeschlagen: {e}")
            logger.info("📊 Aber Chart und Analyse waren erfolgreich!")
            # Nicht als Fehler behandeln - Hauptanalyse war erfolgreich