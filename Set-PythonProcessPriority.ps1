# PowerShell-Skript zum Setzen der Priorität aller laufenden python.exe-Prozesse

# Mögliche Prioritäten
$validPriorities = @("Idle", "BelowNormal", "Normal", "AboveNormal", "High", "RealTime")

# Benutzer nach gewünschter Priorität fragen
Write-Host "Verfügbare Prioritäten: $validPriorities"
$priority = Read-Host "Bitte gib die gewünschte Priorität für python.exe-Prozesse ein"

# Überprüfen, ob die eingegebene Priorität gültig ist
if ($validPriorities -notcontains $priority) {
    Write-Error "Ungültige Priorität '$priority'. Bitte wähle eine aus: $validPriorities"
    exit 1
}

# Alle laufenden python.exe-Prozesse abrufen
$pythonProcesses = Get-Process -Name "python" -ErrorAction SilentlyContinue

if ($pythonProcesses.Count -eq 0) {
    Write-Warning "Keine laufenden python.exe-Prozesse gefunden."
    exit 0
}

# Priorität für jeden python.exe-Prozess setzen
foreach ($process in $pythonProcesses) {
    try {
        $process.PriorityClass = $priority
        Write-Host "Priorität von Prozess ID $($process.Id) ($($process.ProcessName)) auf $priority gesetzt."
    } catch {
        Write-Error "Fehler beim Setzen der Priorität für Prozess ID $($process.Id): $_"
    }
}

Write-Host "Fertig! Priorität für alle python.exe-Prozesse auf $priority gesetzt."