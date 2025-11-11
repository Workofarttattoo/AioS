# Test Results Page Template - Copy & Paste

This is a minimal, reusable template for creating test result pages matching the quantum aesthetic.

## Quick Start Template

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>[PAGE_TITLE] - Ai:oS</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --primary: #00ff88;
            --secondary: #ff00ff;
            --quantum-blue: #00d4ff;
            --quantum-purple: #a855f7;
            --dark: #0a0a0a;
            --darker: #050505;
            --light: #ffffff;
            --accent: #ffaa00;
        }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--darker);
            color: var(--light);
            overflow-x: hidden;
        }

        /* Quantum Background */
        .quantum-bg {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -2;
            background: linear-gradient(180deg, #0a0a1a 0%, #000511 50%, #000000 100%);
        }
        .quantum-grid {
            position: absolute;
            width: 100%;
            height: 100%;
            background-image:
                linear-gradient(rgba(0, 255, 136, 0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 255, 136, 0.02) 1px, transparent 1px);
            background-size: 50px 50px;
            animation: grid-move 10s linear infinite;
        }
        @keyframes grid-move {
            0% { transform: translate(0, 0); }
            100% { transform: translate(50px, 50px); }
        }

        /* Header */
        header {
            padding: 20px 0;
            border-bottom: 1px solid rgba(0, 255, 136, 0.2);
            backdrop-filter: blur(10px);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logo {
            font-size: 24px;
            font-weight: bold;
            background: linear-gradient(45deg, var(--primary), var(--quantum-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            cursor: pointer;
            text-decoration: none;
        }
        nav a {
            color: var(--light);
            text-decoration: none;
            margin-left: 30px;
            font-size: 14px;
            transition: color 0.3s;
        }
        nav a:hover {
            color: var(--primary);
            text-shadow: 0 0 10px var(--primary);
        }

        /* Container */
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 60px 20px;
        }

        /* Headings */
        h1 {
            font-size: 48px;
            margin-bottom: 10px;
            background: linear-gradient(45deg, var(--primary), var(--quantum-blue), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
        }
        .subtitle {
            text-align: center;
            color: rgba(255, 255, 255, 0.7);
            margin-bottom: 60px;
            font-size: 16px;
        }

        /* Cards */
        .card {
            background: rgba(0, 255, 136, 0.05);
            border: 1px solid rgba(0, 255, 136, 0.3);
            border-radius: 12px;
            padding: 25px;
            backdrop-filter: blur(10px);
            transition: all 0.3s;
            margin-bottom: 20px;
        }
        .card:hover {
            background: rgba(0, 255, 136, 0.1);
            border-color: var(--primary);
            box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
            transform: translateY(-5px);
        }

        /* Grid */
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }

        /* Badges */
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .badge-pass {
            background: rgba(0, 255, 136, 0.2);
            color: #00ff88;
            border: 1px solid #00ff88;
        }
        .badge-warn {
            background: rgba(255, 170, 0, 0.2);
            color: #ffaa00;
            border: 1px solid #ffaa00;
        }
        .badge-fail {
            background: rgba(255, 68, 68, 0.2);
            color: #ff4444;
            border: 1px solid #ff4444;
        }

        /* Footer */
        footer {
            border-top: 1px solid rgba(0, 255, 136, 0.2);
            padding: 40px 0;
            margin-top: 80px;
            text-align: center;
            color: rgba(255, 255, 255, 0.5);
        }

        @media (max-width: 768px) {
            h1 { font-size: 32px; }
            nav a { margin-left: 15px; font-size: 12px; }
        }
    </style>
</head>
<body>
    <div class="quantum-bg">
        <div class="quantum-grid"></div>
    </div>

    <!-- Header -->
    <header>
        <div class="header-content">
            <a href="index.html" class="logo">Ai|oS</a>
            <nav>
                <a href="index.html">Home</a>
                <a href="#section1">Section 1</a>
                <a href="#section2">Section 2</a>
            </nav>
        </div>
    </header>

    <!-- Main Content -->
    <div class="container">
        <h1>[PAGE_ICON] [PAGE_TITLE]</h1>
        <p class="subtitle">[PAGE_DESCRIPTION]</p>

        <!-- Section 1 -->
        <section id="section1">
            <h2 style="color: var(--primary); margin: 40px 0 20px 0; font-size: 28px;">📊 [SECTION_TITLE_1]</h2>

            <div class="grid">
                <div class="card">
                    <h3 style="color: var(--quantum-blue); margin-bottom: 12px;">[CARD_TITLE_1]</h3>
                    <p style="font-size: 13px; color: rgba(255,255,255,0.7); line-height: 1.6;">
                        [CARD_CONTENT_1]
                    </p>
                    <span class="badge badge-pass">[STATUS]</span>
                </div>

                <div class="card">
                    <h3 style="color: var(--quantum-blue); margin-bottom: 12px;">[CARD_TITLE_2]</h3>
                    <p style="font-size: 13px; color: rgba(255,255,255,0.7); line-height: 1.6;">
                        [CARD_CONTENT_2]
                    </p>
                    <span class="badge badge-pass">[STATUS]</span>
                </div>

                <div class="card">
                    <h3 style="color: var(--quantum-blue); margin-bottom: 12px;">[CARD_TITLE_3]</h3>
                    <p style="font-size: 13px; color: rgba(255,255,255,0.7); line-height: 1.6;">
                        [CARD_CONTENT_3]
                    </p>
                    <span class="badge badge-warn">[STATUS]</span>
                </div>
            </div>
        </section>

        <!-- Section 2 -->
        <section id="section2">
            <h2 style="color: var(--primary); margin: 40px 0 20px 0; font-size: 28px;">🎯 [SECTION_TITLE_2]</h2>

            <div class="card">
                <h3 style="color: var(--quantum-blue); margin-bottom: 12px;">[CONTENT_TITLE]</h3>
                <p style="font-size: 13px; color: rgba(255,255,255,0.7); line-height: 1.8;">
                    [MAIN_CONTENT_HERE]
                </p>
            </div>
        </section>

        <!-- Updated -->
        <section style="margin-top: 60px;">
            <div style="text-align: center; color: rgba(255,255,255,0.5); font-size: 13px; padding: 20px; background: rgba(0, 255, 136, 0.02); border-radius: 12px; border: 1px solid rgba(0, 255, 136, 0.2);">
                <p>Last Updated: [DATE]</p>
                <p>Version: [VERSION]</p>
            </div>
        </section>
    </div>

    <!-- Footer -->
    <footer>
        <p>&copy; 2025 Joshua Hendricks Cole. All Rights Reserved. PATENT PENDING.</p>
    </footer>
</body>
</html>
```

## How to Use This Template

1. **Copy the code above**
2. **Replace placeholders:**
   - `[PAGE_TITLE]` - Your page title
   - `[PAGE_ICON]` - An emoji for your page
   - `[PAGE_DESCRIPTION]` - Brief description
   - `[SECTION_TITLE_1]`, `[SECTION_TITLE_2]` - Section headings
   - `[CARD_TITLE_1]`, etc. - Card titles
   - `[CARD_CONTENT_1]`, etc. - Card descriptions
   - `[STATUS]` - Use `badge-pass`, `badge-warn`, or `badge-fail`
   - `[DATE]` - Update date
   - `[VERSION]` - Version number

3. **Save as HTML**

4. **Customize colors:**
   - `--primary: #00ff88` (green)
   - `--quantum-blue: #00d4ff` (blue)
   - `--quantum-purple: #a855f7` (purple)
   - `--accent: #ffaa00` (orange)

## Features Included

✅ Quantum background grid
✅ Responsive header with navigation
✅ Beautiful card layout
✅ Hover effects
✅ Badge system (pass/warn/fail)
✅ Mobile friendly
✅ Smooth animations

## Example Customization

```
[PAGE_ICON] → 🧪
[PAGE_TITLE] → Test Results & Compliance
[PAGE_DESCRIPTION] → Comprehensive testing and validation
[SECTION_TITLE_1] → Key Metrics
[CARD_TITLE_1] → Overall Coverage
[CARD_CONTENT_1] → 95% unit & integration test coverage
[STATUS] → badge-pass
```

That's it! You can now create beautiful test pages quickly!
