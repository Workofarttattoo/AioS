#!/bin/bash
# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
#
# Remove /docs/ website from aios CODE repo safely

cd /Users/noone/aios

echo "🔧 REMOVING WEBSITE FROM AIOS CODE REPO"
echo "========================================"
echo ""

# Make sure we're in the right repo
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repo!"
    exit 1
fi

# Check current branch
current_branch=$(git branch --show-current)
echo "Current branch: $current_branch"

# Remove docs from tracking but keep locally
echo ""
echo "📝 Removing /docs/ from git tracking..."
git rm -r --cached docs/

# Add to .gitignore
echo ""
echo "📝 Adding /docs/ to .gitignore..."
echo "docs/" >> .gitignore

# Status
echo ""
echo "✅ Done! Website removed from code repo."
echo ""
echo "📊 Git status:"
git status

echo ""
echo "🎯 NEXT STEPS:"
echo "1. Commit the changes:"
echo "   git add .gitignore"
echo "   git commit -m 'Remove website from code repo - moved to aios-website'"
echo ""
echo "2. Push to GitHub:"
echo "   git push origin $current_branch"
echo ""
echo "✅ The /docs/ folder will still exist locally but won't be in the repo"
