# Ersetze die Zeilen um Zeile 320-350 (Excel Export Bereich) mit:

        # Schritt 8: Excel Export
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_filename = f"enhanced_divergence_analysis_{timestamp}.xlsx"
        
        logger.info(f"📋 Exportiere Ergebnisse: {excel_filename}")
        
        try:
            # VOLLSTÄNDIGE Timezone-Problem beheben
            df_export = df.copy()
            
            # Liste aller problematischen datetime Spalten
            datetime_columns = [
                'date',
                'CBullD_Lower_Low_date_gen', 
                'CBullD_Higher_Low_date_gen',
                'CBullD_Lower_Low_date_neg_MACD', 
                'CBullD_Higher_Low_date_neg_MACD',
                'HBullD_Lower_Low_date',
                'HBullD_Higher_Low_date',
                'BearD_Higher_High_date',
                'BearD_Lower_High_date'
            ]
            
            # Entferne Timezone-Info aus allen datetime Spalten
            for col in datetime_columns:
                if col in df_export.columns:
                    try:
                        # Behandle sowohl datetime als auch object-Spalten mit dates
                        df_export[col] = pd.to_datetime(df_export[col], errors='coerce').dt.tz_localize(None)
                        logger.debug(f"✅ Spalte {col} timezone-free gemacht")
                    except Exception as e:
                        logger.warning(f"⚠️ Konnte Spalte {col} nicht konvertieren: {e}")
            
            # Zusätzlich: Automatische Erkennung aller datetime-Spalten mit Timezone
            for col in df_export.columns:
                try:
                    if str(df_export[col].dtype).startswith('datetime64[ns,') or 'UTC' in str(df_export[col].dtype):
                        df_export[col] = pd.to_datetime(df_export[col]).dt.tz_localize(None)
                        logger.debug(f"🔧 Auto-fix für Spalte {col}")
                except:
                    pass
            
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
                
                # Alle Divergenzen
                all_divergences = []
                for div_col in ['CBullD_gen', 'CBullD_neg_MACD', 'HBullD', 'BearD']:
                    if div_col in df.columns:
                        div_data = df[df[div_col] == 1].copy()
                        if not div_data.empty:
                            div_data['Divergence_Type'] = div_col
                            # Auch hier Timezone entfernen
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
            logger.info("📊 Aber Analyse wurde erfolgreich durchgeführt - Chart ist verfügbar!")
            # Setze Erfolg trotzdem, da die Hauptanalyse funktioniert hat
            pass