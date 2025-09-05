import os
import pandas as pd
import numpy as np
import base64, io
import matplotlib.pyplot as plt

def run_portfolio_allocation(paths, out_xlsx: str, out_html: str):
    """Scaffold: load multiple CSVs, compute a simple return correlation matrix.
    Assumes columns contain 'date' and 'close'. Writes minimal outputs.
    """
    series = {}
    for p in paths:
        try:
            df = pd.read_csv(p)
            if 'date' in df.columns and 'close' in df.columns:
                s = pd.to_datetime(df['date'])
                r = pd.Series(df['close']).pct_change().rename(os.path.basename(p))
                r.index = s
                series[os.path.basename(p)] = r
        except Exception:
            continue
    aligned = None
    if not series:
        mat = pd.DataFrame({'Note':['No compatible files loaded']})
    else:
        # Build combined returns frame and keep only overlapping periods
        df_all = pd.DataFrame(series).dropna(how='all')
        aligned = df_all.dropna()
        mat = aligned.corr()
    # Per-asset simple cumulative returns (profit)
    asset_perf = pd.DataFrame()
    indiv_perf = pd.DataFrame()
    if aligned is not None and not aligned.empty:
        rets = aligned.fillna(0.0)
        eq = (1.0 + rets).cumprod()
        last = eq.tail(1).T
        last.columns = ['Final_Equity']
        last['PnL_%'] = (last['Final_Equity'] - 1.0) * 100.0
        asset_perf = last.sort_values('PnL_%', ascending=False)
    else:
        # Fall back to individual series performance even without full overlap
        rows = []
        for name, s in series.items():
            try:
                ss = s.dropna()
                if ss.empty:
                    continue
                eq = (1.0 + ss).cumprod()
                final = float(eq.iloc[-1])
                pnl = (final - 1.0) * 100.0
                rows.append({'Asset': name, 'Final_Equity': final, 'PnL_%': pnl})
            except Exception:
                pass
        if rows:
            indiv_perf = pd.DataFrame(rows).sort_values('PnL_%', ascending=False)

    # Allocation variations: discrete weights per asset, evaluate portfolio PnL
    alloc_table = pd.DataFrame()
    best_curve_img64 = ''
    if aligned is not None and not aligned.empty and aligned.shape[1] >= 2:
        choices = [0.1, 0.25, 0.5, 0.75]
        names = list(aligned.columns)
        rng = np.random.default_rng(123)
        n_assets = len(names)
        max_samples = 500
        samples = []
        for _ in range(max_samples):
            w = rng.choice(choices, size=n_assets)
            w = w / np.sum(w)
            samples.append(w)
        samples = np.array(samples)
        rets = aligned.fillna(0.0)
        port_rows = []
        for w in samples:
            w_series = pd.Series(w, index=names)
            port_ret = (rets * w_series).sum(axis=1)
            port_eq = (1.0 + port_ret).cumprod()
            final = float(port_eq.iloc[-1])
            pnl = (final - 1.0) * 100.0
            vol = float(port_ret.std()) if port_ret.std() > 0 else np.nan
            sharpe = float(port_ret.mean()/port_ret.std()) if vol and vol>0 else np.nan
            port_rows.append({'Weights': '; '.join(f'{n}={w_series[n]:.2f}' for n in names), 'Final_Equity': final, 'PnL_%': pnl, 'Sharpe_surrogate': sharpe})
        alloc_table = pd.DataFrame(port_rows).sort_values('PnL_%', ascending=False).head(10)
        # Best curve chart
        try:
            best_weights = samples[np.argmax([r['PnL_%'] for r in port_rows])]
            w_series = pd.Series(best_weights, index=names)
            port_ret = (rets * w_series).sum(axis=1)
            port_eq = (1.0 + port_ret).cumprod()
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.plot(port_eq.index, port_eq.values, color='#2c3e50', lw=1.2)
            ax.set_title('Beste Portfolio‑Equity (Diskrete Gewichte)')
            ax.tick_params(axis='x', labelrotation=30)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120)
            plt.close(fig)
            best_curve_img64 = base64.b64encode(buf.getvalue()).decode('ascii')
        except Exception:
            best_curve_img64 = ''
    os.makedirs(os.path.dirname(out_xlsx) or '.', exist_ok=True)
    try:
        with pd.ExcelWriter(out_xlsx, engine='openpyxl') as w:
            mat.to_excel(w, sheet_name='Correlation')
            if not asset_perf.empty:
                asset_perf.to_excel(w, sheet_name='Asset_Perf', index=True)
            if not alloc_table.empty:
                alloc_table.to_excel(w, sheet_name='Allocations', index=False)
    except Exception:
        mat.to_csv(out_xlsx.replace('.xlsx', '.csv'))
    # Heatmap image
    img64 = ''
    try:
        if not mat.empty and isinstance(mat, pd.DataFrame) and mat.shape[0] > 1:
            fig, ax = plt.subplots(figsize=(4 + 0.2*mat.shape[1], 3 + 0.2*mat.shape[0]))
            cax = ax.imshow(mat.values, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index, fontsize=8)
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title('Korrelation zwischen Assets')
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120)
            plt.close(fig)
            img64 = base64.b64encode(buf.getvalue()).decode('ascii')
    except Exception:
        img64 = ''

    with open(out_html, 'w', encoding='utf-8') as f:
        f.write('<html><body>')
        f.write('<h3>Portfolio Layer</h3>')
        f.write('<ul>')
        f.write('<li><b>Voraussetzungen:</b> Mehrere Asset‑Dateien mit Spalten date und close; genügend zeitliche Überlappung für gemeinsame Renditen.</li>')
        f.write('<li><b>Was wird analysiert?</b> Rendite‑Korrelationen zwischen Assets (einfacher Überblick) und einfache Portfoliovarianten mit diskreten Gewichten.</li>')
        f.write('<li><b>Wofür verwenden?</b> Portfoliogewichtung (z. B. Risiko‑Parität, Vol‑Targeting) und Klumpenrisiken erkennen.</li>')
        f.write('<li><b>Warum wichtig?</b> Gering korrelierte Komponenten senken Drawdowns bei ähnlichen Erträgen.</li>')
        f.write('<li><b>Leer oder identische Ergebnisse?</b> Wenn kaum Überlappung zwischen Zeitreihen besteht, sind gemeinsame Renditen leer (oder nahe 0) und Final‑Equity kann für viele Varianten gleich wirken. In diesem Fall werden Einzel‑Asset‑Ergebnisse separat ausgewiesen.</li>')
        f.write('</ul>')
        if img64:
            f.write("<h4>Korrelationen</h4><img src='data:image/png;base64,%s' />" % img64)
        f.write('<h4>Matrix</h4><pre>' + mat.to_string() + '</pre>')
        # Diagnostic about overlap
        try:
            if aligned is not None:
                f.write(f"<p>Überlappende Zeilen: {0 if aligned is None else aligned.shape[0]}</p>")
        except Exception:
            pass
        if not asset_perf.empty:
            f.write('<h4>Per‑Asset Ertrag</h4>')
            f.write(asset_perf.round(2).to_html())
        elif not indiv_perf.empty:
            f.write('<h4>Per‑Asset Ertrag (individuell, ohne gemeinsame Überlappung)</h4>')
            f.write(indiv_perf.round(2).to_html(index=False))
        if not alloc_table.empty:
            f.write('<h4>Top Portfoliogewichte (PnL)</h4>')
            f.write(alloc_table.round(3).to_html(index=False))
            if best_curve_img64:
                f.write("<h4>Beste Portfolio‑Equity</h4><img src='data:image/png;base64,%s' />" % best_curve_img64)
        f.write('</body></html>')
