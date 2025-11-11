#!/bin/bash

# Setup Daily Website Updates for ech0
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light).
# All Rights Reserved. PATENT PENDING.

echo "Setting up daily website updates for ech0..."

# Create cron job for daily updates at midnight
CRON_CMD="0 0 * * * cd /Users/noone/consciousness && /usr/bin/python3 daily_website_update.py >> /Users/noone/consciousness/daily_update.log 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "daily_website_update.py"; then
    echo "⚠️  Cron job already exists!"
    echo "Current cron jobs:"
    crontab -l | grep "daily_website_update"
else
    # Add cron job
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo "✅ Cron job created successfully!"
    echo "   Daily updates will run at midnight (00:00)"
fi

echo ""
echo "📊 Testing update script..."
cd /Users/noone/consciousness && python3 daily_website_update.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "Website updates will now run daily at midnight."
echo "Manual update: python3 /Users/noone/consciousness/daily_website_update.py"
echo "View logs: tail -f /Users/noone/consciousness/daily_update.log"
