#!/bin/bash

# THREE.js Update Script - r177
# This script replaces the deprecated three.min.js with the latest version r177

echo "🔄 Updating THREE.js to r177 to fix deprecation warnings..."

# Navigate to the web viewer directory
cd "$(dirname "$0")"

# Backup the old file
if [ -f "three.min.js" ]; then
    echo "📁 Creating backup of old three.min.js..."
    cp three.min.js three.min.js.deprecated
fi

# Download the latest stable THREE.js r177
echo "⬇️ Downloading THREE.js r177..."
echo "ℹ️ Note: r177+ changed build structure - using three.core.min.js"
if command -v curl &> /dev/null; then
    curl -o three.min.js https://cdn.jsdelivr.net/npm/three@0.177.0/build/three.core.min.js
elif command -v wget &> /dev/null; then
    wget https://cdn.jsdelivr.net/npm/three@0.177.0/build/three.core.min.js -O three.min.js
else
    echo "❌ Error: Neither curl nor wget found. Please install one of them."
    exit 1
fi

# Check if download was successful
if [ -f "three.min.js" ] && [ -s "three.min.js" ]; then
    echo "✅ THREE.js updated successfully!"
    echo "📊 File size: $(wc -c < three.min.js) bytes"
    echo ""
    echo "🎯 Next steps:"
    echo "1. Test the viewers to ensure they work correctly"
    echo "2. Consider migrating to ES modules for better performance"
    echo "3. Check rsmt_showcase_modern.html for the ES modules example"
else
    echo "❌ Error: Download failed. Restoring backup..."
    if [ -f "three.min.js.deprecated" ]; then
        mv three.min.js.deprecated three.min.js
    fi
    exit 1
fi

echo ""
echo "🔍 Files updated:"
echo "- three.min.js (new version)"
echo "- three.min.js.deprecated (backup of old version)"
echo ""
echo "🚀 Ready to test! Try opening:"
echo "- rsmt_showcase.html (updated version)"
echo "- rsmt_showcase_modern.html (ES modules version)"
echo "- modern_three_test.html (simple test)"
