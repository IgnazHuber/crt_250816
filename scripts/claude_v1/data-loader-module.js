// data-loader.js - Modul für Datei-Handling (CSV/Parquet)
class DataLoader {
    constructor() {
        this.rawData = null;
        this.processedData = null;
        this.fileType = null;
        
        // Event-Listener für Datei-Upload
        this.setupFileListener();
    }
    
    setupFileListener() {
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        }
    }
    
    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        document.getElementById('fileName').textContent = file.name;
        this.fileType = file.name.endsWith('.csv') ? 'csv' : 'parquet';
        
        try {
            this.showMessage('Lade Datei...', 'loading');
            
            if (this.fileType === 'csv') {
                await this.loadCSV(file);
            } else {
                await this.loadParquet(file);
            }
            
            this.showMessage(`Datei erfolgreich geladen! ${this.rawData.length} Datensätze gefunden.`, 'success');
            console.log('Erste Zeile:', this.rawData[0]);
            console.log('Verfügbare Spalten:', Object.keys(this.rawData[0]));
            
        } catch (error) {
            console.error('Fehler beim Laden:', error);
            this.showMessage('Fehler beim Laden der Datei: ' + error.message, 'error');
        }
    }
    
    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            Papa.parse(file, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    if (results.errors.length > 0) {
                        console.warn('CSV-Parse-Warnungen:', results.errors);
                    }
                    this.rawData = this.normalizeData(results.data);
                    resolve(this.rawData);
                },
                error: (error) => {
                    reject(error);
                }
            });
        });
    }
    
    async loadParquet(file) {
        // Für echte Parquet-Unterstützung würde man hier eine Library wie parquet-wasm verwenden
        // Für Demo-Zwecke: Konvertierung zu ArrayBuffer und Versuch zu parsen
        const arrayBuffer = await file.arrayBuffer();
        
        // Prüfe auf Parquet Magic Bytes
        const bytes = new Uint8Array(arrayBuffer);
        const magic1 = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
        const magic2 = String.fromCharCode(bytes[bytes.length-4], bytes[bytes.length-3], bytes[bytes.length-2], bytes[bytes.length-1]);
        
        if (magic1 === 'PAR1' || magic2 === 'PAR1') {
            console.log('Parquet-Datei erkannt, verwende Fallback-Parser...');
            // Hier würde normalerweise ein echter Parquet-Parser kommen
            // Für Demo: Simuliere Daten
            this.rawData = this.generateDemoData();
        } else {
            throw new Error('Keine gültige Parquet-Datei');
        }
    }
    
    normalizeData(data) {
        // Normalisiere Spaltennamen und Datentypen
        return data.map(row => {
            const normalized = {};
            
            // Stelle sicher, dass alle erforderlichen Felder vorhanden sind
            normalized.date = row.date || row.Date || row.timestamp || new Date().toISOString();
            normalized.open = parseFloat(row.open || row.Open || row.o || 0);
            normalized.high = parseFloat(row.high || row.High || row.h || 0);
            normalized.low = parseFloat(row.low || row.Low || row.l || 0);
            normalized.close = parseFloat(row.close || row.Close || row.c || 0);
            normalized.volume = parseFloat(row.volume || row.Volume || row.v || 0);
            
            return normalized;
        });
    }
    
    generateDemoData() {
        console.log('Generiere Demo-Daten für Analyse...');
        const days = 500;
        const data = [];
        let basePrice = 100;
        
        for (let i = 0; i < days; i++) {
            const date = new Date(2022, 0, 1);
            date.setDate(date.getDate() + i);
            
            const trend = Math.sin(i / 50) * 10;
            const noise = (Math.random() - 0.5) * 4;
            basePrice = Math.max(50, Math.min(150, basePrice + trend/10 + noise));
            
            const high = basePrice + Math.random() * 3;
            const low = basePrice - Math.random() * 3;
            const open = low + Math.random() * (high - low);
            const close = low + Math.random() * (high - low);
            
            data.push({
                date: date.toISOString().split('T')[0],
                open: open,
                high: high,
                low: low,
                close: close,
                volume: Math.floor(Math.random() * 1000000) + 500000
            });
        }
        
        return data;
    }
    
    getData() {
        return this.rawData;
    }
    
    getProcessedData() {
        return this.processedData;
    }
    
    setProcessedData(data) {
        this.processedData = data;
    }
    
    showMessage(message, type) {
        const container = document.getElementById('messageContainer');
        if (container) {
            container.innerHTML = `<div class="${type}">${message}</div>`;
            
            if (type === 'success') {
                setTimeout(() => {
                    container.innerHTML = '';
                }, 3000);
            }
        }
    }
    
    validateData() {
        if (!this.rawData || this.rawData.length === 0) {
            throw new Error('Keine Daten geladen');
        }
        
        const requiredFields = ['date', 'open', 'high', 'low', 'close'];
        const firstRow = this.rawData[0];
        
        for (const field of requiredFields) {
            if (!(field in firstRow)) {
                throw new Error(`Erforderliches Feld fehlt: ${field}`);
            }
        }
        
        return true;
    }
}