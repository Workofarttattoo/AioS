#!/bin/bash
# Ai|oS Desktop Launcher Script
# Opens the Ai|oS launcher in fullscreen mode

LAUNCHER_PATH="/Users/noone/aios/web/aios_launcher.html"

echo "🐺 Starting Ai|oS Desktop Launcher..."

# Try to open with Chrome app mode (cleanest fullscreen experience)
if command -v open &> /dev/null; then
    # macOS
    if [ -d "/Applications/Google Chrome.app" ]; then
        open -a "Google Chrome" --args --app="file://$LAUNCHER_PATH" --start-fullscreen
    elif [ -d "/Applications/Firefox.app" ]; then
        open -a "Firefox" "file://$LAUNCHER_PATH"
    else
        # Fallback to default browser
        open "file://$LAUNCHER_PATH"
    fi
else
    # Linux
    if command -v google-chrome &> /dev/null; then
        google-chrome --app="file://$LAUNCHER_PATH" --start-fullscreen &
    elif command -v firefox &> /dev/null; then
        firefox "file://$LAUNCHER_PATH" &
    else
        xdg-open "file://$LAUNCHER_PATH" &
    fi
fi

echo "✅ Launcher opened"
echo "Press Ctrl+Q or Cmd+Q to exit fullscreen"
