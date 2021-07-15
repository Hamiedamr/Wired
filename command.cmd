# Source: https://stackoverflow.com/a/43317244
$path = ".\wired-key.pem"
# Reset to remove explict permissions
icacls.exe $path /reset
# Give current user explicit read-permission
icacls.exe $path /GRANT:R "$($env:hamed):(R)"
# Disable inheritance and remove inherited permissions
icacls.exe $path /inheritance:r