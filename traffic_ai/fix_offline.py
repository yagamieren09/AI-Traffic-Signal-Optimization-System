"""
fix_offline.py
Run this ONCE from inside the traffic_ai folder.
It downloads React, ReactDOM, and Recharts to a local libs/ folder
and patches index.html to use local files instead of CDN.

Usage:
    cd traffic_ai
    python fix_offline.py
"""

import urllib.request
import os
import shutil

LIBS_DIR = os.path.join("frontend", "dist", "libs")
HTML_PATH = os.path.join("frontend", "dist", "index.html")

LIBS = [
    (
        "https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js",
        "react.min.js"
    ),
    (
        "https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js",
        "react-dom.min.js"
    ),
    (
        "https://cdnjs.cloudflare.com/ajax/libs/recharts/2.8.0/Recharts.js",
        "recharts.min.js"
    ),
]

def download_libs():
    os.makedirs(LIBS_DIR, exist_ok=True)
    all_ok = True
    for url, filename in LIBS:
        dest = os.path.join(LIBS_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 10000:
            print(f"  [SKIP] {filename} already exists")
            continue
        print(f"  Downloading {filename} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            size = os.path.getsize(dest)
            print(f"OK ({size // 1024} KB)")
        except Exception as e:
            print(f"FAILED: {e}")
            all_ok = False
    return all_ok

def patch_html():
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    # Check if already patched
    if "libs/react.min.js" in html:
        print("  [SKIP] index.html already patched")
        return

    # Back up original
    shutil.copy(HTML_PATH, HTML_PATH + ".bak")

    # Replace CDN links with local paths
    html = html.replace(
        'src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"',
        'src="libs/react.min.js"'
    )
    html = html.replace(
        'src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"',
        'src="libs/react-dom.min.js"'
    )
    html = html.replace(
        'src="https://cdnjs.cloudflare.com/ajax/libs/recharts/2.8.0/Recharts.js"',
        'src="libs/recharts.min.js"'
    )

    # Remove Google Fonts (works offline with fallback fonts)
    html = html.replace(
        '<link rel="preconnect" href="https://fonts.googleapis.com">',
        ''
    )
    lines = html.split('\n')
    lines = [l for l in lines if 'fonts.googleapis.com/css2' not in l]
    html = '\n'.join(lines)

    # Add offline font fallbacks in the CSS variables
    html = html.replace(
        "'Share Tech Mono', monospace",
        "Consolas, 'Courier New', monospace"
    )
    html = html.replace(
        "'Barlow', sans-serif",
        "Segoe UI, Arial, sans-serif"
    )
    html = html.replace(
        "'Barlow Condensed', sans-serif",
        "Segoe UI, Arial, sans-serif"
    )

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print("  index.html patched to use local libraries")

def main():
    print("\n=== Traffic AI — Offline Fix ===\n")

    if not os.path.exists(HTML_PATH):
        print(f"ERROR: Cannot find {HTML_PATH}")
        print("Make sure you are running this from INSIDE the traffic_ai folder.")
        print("  cd traffic_ai")
        print("  python fix_offline.py")
        input("\nPress Enter to exit...")
        return

    print("Step 1: Downloading JavaScript libraries locally...")
    ok = download_libs()

    if not ok:
        print("\nSome downloads failed. Check your internet connection and try again.")
        input("\nPress Enter to exit...")
        return

    print("\nStep 2: Patching index.html to use local libraries...")
    patch_html()

    print("\n✅ All done! Now run the server:")
    print("   python server.py")
    print("   Then open: http://localhost:5000\n")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
