# PowerShell-Skript zum regelmäßigen Setzen der Priorität von python.exe und RuntimeBroker.exe auf Niedrig

# Mögliche Prioritäten (nur für Referenz, da wir fest BelowNormal verwenden)
$validPriorities = @("Idle", "BelowNormal", "Normal", "AboveNormal", "High", "RealTime")
$targetPriority = "Idle" #"BelowNormal"

Write-Host "Skript gestartet. Setzt alle python.exe und RuntimeBroker.exe Prozesse jede Minute auf Priorität $targetPriority."
Write-Host "Drücke Ctrl+C, um das Skript zu beenden."

# Endlosschleife für regelmäßige Ausführung
while ($true) {
    # Zeitstempel für den aktuellen Durchlauf
    Write-Host "Durchlauf um $(Get-Date -Format 'HH:mm:ss')"

    # Alle laufenden python.exe und RuntimeBroker.exe Prozesse abrufen
    $processes = Get-Process -Name "python", "RuntimeBroker", "pwsh" -ErrorAction SilentlyContinue

    if ($processes.Count -eq 0) {
        Write-Warning "Keine laufenden python.exe oder RuntimeBroker.exe Prozesse gefunden."
    } else {
        # Priorität für jeden Prozess setzen
        foreach ($process in $processes) {
            try {
                $process.PriorityClass = $targetPriority
                Write-Host "Priorität von Prozess ID $($process.Id) ($($process.ProcessName)) auf $targetPriority gesetzt."
            } catch {
                Write-Error "Fehler beim Setzen der Priorität für Prozess ID $($process.Id): $_"
            }
        }
    }

    # Warte 22 Sekunden bis zum nächsten Durchlauf
    Start-Sleep -Seconds 22
}

Write-Host "Skript beendet."