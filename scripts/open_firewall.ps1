# Add a Windows Firewall inbound rule allowing TCP 48763 (the ocr-agent service).
# Must be run as Administrator (UAC).

$ErrorActionPreference = 'Stop'
$RULE = 'ocr-agent (TCP 48763)'
$PORT = 48763

# Idempotent: remove an existing same-named rule first.
$existing = Get-NetFirewallRule -DisplayName $RULE -ErrorAction SilentlyContinue
if ($existing) {
    Write-Output "Rule '$RULE' already exists — removing old one first."
    Remove-NetFirewallRule -DisplayName $RULE
}

# Create the inbound rule: allow TCP 48763 from any remote address.
New-NetFirewallRule `
    -DisplayName $RULE `
    -Description 'Inbound TCP 48763 for the ocr-agent uvicorn service (LAN access).' `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort $PORT `
    -Profile Any `
    -Enabled True | Out-Null

Write-Output "FIREWALL_RULE_CREATED: $RULE (TCP $PORT, inbound, any profile)"

# Verify it's there.
$check = Get-NetFirewallRule -DisplayName $RULE -ErrorAction SilentlyContinue
if ($check) {
    Write-Output ("VERIFIED: enabled=" + $check.Enabled + " direction=" + $check.Direction + " action=" + $check.Action)
} else {
    Write-Output "VERIFICATION FAILED: rule not found after creation"
}
