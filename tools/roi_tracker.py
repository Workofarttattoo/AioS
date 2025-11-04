#!/usr/bin/env python3
"""
ROI Tracking System for Security Tools
Integrates email capture, conversion tracking, and analytics

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.
"""

import json
import time
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class ConversionEvent:
    """Represents a conversion event"""
    timestamp: str
    tool_name: str
    event_type: str  # 'email_capture', 'flowstate_click', 'trial_start', 'paid_conversion'
    user_id: str
    variant: Optional[str] = None
    metadata: Optional[Dict] = None

    def to_dict(self):
        return asdict(self)

class ROITracker:
    """
    Tracks ROI metrics for security tools
    """

    def __init__(self, data_dir: str = "./roi_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # High-traffic tools get FlowState CTAs
        self.high_traffic_tools = [
            'AuroraScan', 'SpectraTrace', 'MythicKey',
            'CipherSpear', 'SkyBreaker'
        ]

        # A/B test variants
        self.variants = {
            'A': {
                'headline': 'Enterprise Security Testing',
                'value_prop': 'Trusted by Fortune 500',
                'cta': 'Start Free Trial'
            },
            'B': {
                'headline': '10x Faster Scans',
                'value_prop': 'Results in minutes',
                'cta': 'Speed Up Now'
            },
            'C': {
                'headline': 'Save $50K/year',
                'value_prop': '80% cost reduction',
                'cta': 'Calculate Savings'
            }
        }

        # Load or initialize metrics
        self.metrics_file = self.data_dir / "metrics.json"
        self.load_metrics()

    def load_metrics(self):
        """Load existing metrics or initialize new ones"""
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
        else:
            self.metrics = {
                'total_conversions': 0,
                'email_captures': 0,
                'flowstate_clicks': 0,
                'trial_signups': 0,
                'paid_conversions': 0,
                'revenue': 0,
                'tool_metrics': {},
                'variant_performance': {
                    'A': {'views': 0, 'conversions': 0},
                    'B': {'views': 0, 'conversions': 0},
                    'C': {'views': 0, 'conversions': 0}
                },
                'daily_metrics': {}
            }

    def save_metrics(self):
        """Save metrics to disk"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def generate_user_id(self, email: str = None) -> str:
        """Generate a unique user ID"""
        if email:
            return hashlib.md5(email.encode()).hexdigest()[:12]
        return hashlib.md5(str(time.time()).encode()).hexdigest()[:12]

    def track_event(self, event: ConversionEvent):
        """Track a conversion event"""
        # Update general metrics
        self.metrics['total_conversions'] += 1

        if event.event_type == 'email_capture':
            self.metrics['email_captures'] += 1
        elif event.event_type == 'flowstate_click':
            self.metrics['flowstate_clicks'] += 1
        elif event.event_type == 'trial_start':
            self.metrics['trial_signups'] += 1
        elif event.event_type == 'paid_conversion':
            self.metrics['paid_conversions'] += 1
            self.metrics['revenue'] += event.metadata.get('amount', 47)

        # Update tool-specific metrics
        if event.tool_name not in self.metrics['tool_metrics']:
            self.metrics['tool_metrics'][event.tool_name] = {
                'views': 0,
                'email_captures': 0,
                'flowstate_clicks': 0,
                'conversions': 0,
                'revenue': 0
            }

        tool_metrics = self.metrics['tool_metrics'][event.tool_name]
        tool_metrics['views'] += 1

        if event.event_type == 'email_capture':
            tool_metrics['email_captures'] += 1
        elif event.event_type == 'flowstate_click':
            tool_metrics['flowstate_clicks'] += 1
        elif event.event_type == 'paid_conversion':
            tool_metrics['conversions'] += 1
            tool_metrics['revenue'] += event.metadata.get('amount', 47)

        # Update variant performance if applicable
        if event.variant:
            self.metrics['variant_performance'][event.variant]['views'] += 1
            if event.event_type in ['email_capture', 'trial_start', 'paid_conversion']:
                self.metrics['variant_performance'][event.variant]['conversions'] += 1

        # Update daily metrics
        today = datetime.now().strftime('%Y-%m-%d')
        if today not in self.metrics['daily_metrics']:
            self.metrics['daily_metrics'][today] = {
                'conversions': 0,
                'revenue': 0,
                'email_captures': 0
            }

        self.metrics['daily_metrics'][today]['conversions'] += 1
        if event.event_type == 'email_capture':
            self.metrics['daily_metrics'][today]['email_captures'] += 1
        if event.event_type == 'paid_conversion':
            self.metrics['daily_metrics'][today]['revenue'] += event.metadata.get('amount', 47)

        # Save metrics
        self.save_metrics()

        # Log event
        self.log_event(event)

    def log_event(self, event: ConversionEvent):
        """Log event to file for analysis"""
        log_file = self.data_dir / f"events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(event.to_dict()) + '\n')

    def get_ab_test_winner(self) -> Tuple[str, float]:
        """Determine the winning A/B test variant"""
        best_variant = 'A'
        best_rate = 0

        for variant, data in self.metrics['variant_performance'].items():
            if data['views'] > 0:
                conversion_rate = data['conversions'] / data['views']
                if conversion_rate > best_rate:
                    best_rate = conversion_rate
                    best_variant = variant

        return best_variant, best_rate * 100

    def get_top_converting_tools(self, limit: int = 5) -> List[Dict]:
        """Get the top converting tools"""
        tool_conversions = []

        for tool, metrics in self.metrics['tool_metrics'].items():
            if metrics['views'] > 0:
                conversion_rate = metrics['conversions'] / metrics['views']
                tool_conversions.append({
                    'tool': tool,
                    'conversions': metrics['conversions'],
                    'rate': conversion_rate * 100,
                    'revenue': metrics['revenue'],
                    'is_high_traffic': tool in self.high_traffic_tools
                })

        # Sort by conversion rate
        tool_conversions.sort(key=lambda x: x['rate'], reverse=True)
        return tool_conversions[:limit]

    def get_conversion_funnel(self) -> Dict:
        """Get conversion funnel metrics"""
        total_views = sum(m['views'] for m in self.metrics['tool_metrics'].values())

        if total_views == 0:
            return {
                'visits': 0,
                'email_capture_rate': 0,
                'flowstate_ctr': 0,
                'trial_rate': 0,
                'paid_rate': 0
            }

        return {
            'visits': total_views,
            'email_capture_rate': (self.metrics['email_captures'] / total_views) * 100,
            'flowstate_ctr': (self.metrics['flowstate_clicks'] / total_views) * 100,
            'trial_rate': (self.metrics['trial_signups'] / total_views) * 100,
            'paid_rate': (self.metrics['paid_conversions'] / total_views) * 100
        }

    def generate_email_capture_html(self, tool_name: str) -> str:
        """Generate HTML for email capture form"""
        is_high_traffic = tool_name in self.high_traffic_tools

        html = f"""
        <div class="roi-email-capture" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h3 style="color: white; margin-bottom: 10px;">🚀 Get {tool_name} Pro Features Free</h3>
            <p style="color: rgba(255,255,255,0.9); margin-bottom: 15px;">
                Join 10,000+ security professionals using {tool_name}
            </p>
            <form onsubmit="trackEmailCapture(event, '{tool_name}')" style="display: flex; gap: 10px;">
                <input type="email" placeholder="Enter your email" required
                       style="flex: 1; padding: 10px; border-radius: 5px; border: none;">
                <button type="submit" style="padding: 10px 20px; background: white; color: #667eea;
                        border: none; border-radius: 5px; font-weight: bold; cursor: pointer;">
                    Get Free Access
                </button>
            </form>
        """

        if is_high_traffic:
            html += f"""
            <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 5px;">
                <span style="background: #fbbf24; color: black; padding: 2px 8px; border-radius: 3px;
                      font-size: 12px; font-weight: bold;">FLOWSTATE ENTERPRISE</span>
                <p style="color: white; margin: 10px 0;">
                    ⚡ Unlock unlimited scans, API access, and priority support
                </p>
                <a href="#" onclick="trackFlowStateClick('{tool_name}')"
                   style="display: inline-block; padding: 8px 16px; background: #fbbf24;
                   color: black; text-decoration: none; border-radius: 5px; font-weight: bold;">
                    Start Free Trial →
                </a>
            </div>
            """

        html += """
        </div>
        <script>
        function trackEmailCapture(e, tool) {
            e.preventDefault();
            // Send to analytics
            if (typeof gtag !== 'undefined') {
                gtag('event', 'email_capture', {
                    'tool_name': tool,
                    'timestamp': new Date().toISOString()
                });
            }
            console.log('Email captured for tool:', tool);
            alert('Thank you! Check your email for access instructions.');
        }

        function trackFlowStateClick(tool) {
            // Send to analytics
            if (typeof gtag !== 'undefined') {
                gtag('event', 'flowstate_click', {
                    'tool_name': tool,
                    'timestamp': new Date().toISOString()
                });
            }
            console.log('FlowState CTA clicked for tool:', tool);
        }
        </script>
        """

        return html

    def simulate_traffic(self, days: int = 7):
        """Simulate traffic for testing/demo purposes"""
        tools = list(self.metrics['tool_metrics'].keys()) or [
            'AuroraScan', 'SpectraTrace', 'MythicKey', 'CipherSpear',
            'SkyBreaker', 'NemesisHydra', 'ObsidianHunt', 'VectorFlux'
        ]

        for day in range(days):
            date = (datetime.now() - timedelta(days=days-day-1))

            for tool in tools:
                # Simulate views
                views = random.randint(50, 500)

                for _ in range(views):
                    # Choose random variant
                    variant = random.choice(['A', 'B', 'C'])

                    # Simulate conversion funnel
                    user_id = self.generate_user_id()

                    # 35% email capture rate
                    if random.random() < 0.35:
                        event = ConversionEvent(
                            timestamp=date.isoformat(),
                            tool_name=tool,
                            event_type='email_capture',
                            user_id=user_id,
                            variant=variant
                        )
                        self.track_event(event)

                        # 25% of email captures click FlowState
                        if random.random() < 0.25 and tool in self.high_traffic_tools:
                            event = ConversionEvent(
                                timestamp=date.isoformat(),
                                tool_name=tool,
                                event_type='flowstate_click',
                                user_id=user_id,
                                variant=variant
                            )
                            self.track_event(event)

                            # 50% of FlowState clicks start trial
                            if random.random() < 0.5:
                                event = ConversionEvent(
                                    timestamp=date.isoformat(),
                                    tool_name=tool,
                                    event_type='trial_start',
                                    user_id=user_id,
                                    variant=variant
                                )
                                self.track_event(event)

                                # 40% of trials convert to paid
                                if random.random() < 0.4:
                                    amount = random.choice([47, 97, 197])
                                    event = ConversionEvent(
                                        timestamp=date.isoformat(),
                                        tool_name=tool,
                                        event_type='paid_conversion',
                                        user_id=user_id,
                                        variant=variant,
                                        metadata={'amount': amount}
                                    )
                                    self.track_event(event)

    def get_dashboard_data(self) -> Dict:
        """Get all data for dashboard display"""
        return {
            'summary': {
                'total_conversions': self.metrics['total_conversions'],
                'email_captures': self.metrics['email_captures'],
                'flowstate_clicks': self.metrics['flowstate_clicks'],
                'trial_signups': self.metrics['trial_signups'],
                'paid_conversions': self.metrics['paid_conversions'],
                'total_revenue': f"${self.metrics['revenue']:,.2f}",
                'avg_revenue_per_user': f"${self.metrics['revenue'] / max(self.metrics['paid_conversions'], 1):.2f}"
            },
            'top_tools': self.get_top_converting_tools(),
            'ab_test': {
                'winner': self.get_ab_test_winner(),
                'variants': self.metrics['variant_performance']
            },
            'funnel': self.get_conversion_funnel(),
            'daily_metrics': self.metrics['daily_metrics']
        }

def main():
    """Demo the ROI tracking system"""
    print("=" * 60)
    print("ROI TRACKING SYSTEM FOR SECURITY TOOLS")
    print("=" * 60)

    tracker = ROITracker()

    # Simulate some traffic
    print("\n📊 Simulating 7 days of traffic...")
    tracker.simulate_traffic(days=7)

    # Get dashboard data
    data = tracker.get_dashboard_data()

    print("\n📈 PERFORMANCE SUMMARY:")
    print("-" * 40)
    for key, value in data['summary'].items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    print("\n🏆 TOP CONVERTING TOOLS:")
    print("-" * 40)
    for tool in data['top_tools']:
        traffic = "⚡ HIGH TRAFFIC" if tool['is_high_traffic'] else ""
        print(f"{tool['tool']}: {tool['rate']:.1f}% conversion, ${tool['revenue']} revenue {traffic}")

    print("\n🧪 A/B TEST RESULTS:")
    print("-" * 40)
    winner, rate = data['ab_test']['winner']
    print(f"Winning Variant: {winner} ({rate:.1f}% conversion rate)")

    print("\n🎯 CONVERSION FUNNEL:")
    print("-" * 40)
    funnel = data['funnel']
    print(f"Visits: {funnel['visits']}")
    print(f"Email Capture: {funnel['email_capture_rate']:.1f}%")
    print(f"FlowState CTR: {funnel['flowstate_ctr']:.1f}%")
    print(f"Trial Rate: {funnel['trial_rate']:.1f}%")
    print(f"Paid Conversion: {funnel['paid_rate']:.1f}%")

    # Generate sample email capture HTML
    print("\n📧 SAMPLE EMAIL CAPTURE HTML:")
    print("-" * 40)
    sample_html = tracker.generate_email_capture_html('AuroraScan')
    print("HTML generated for high-traffic tool (see roi_data/sample_capture.html)")

    with open('roi_data/sample_capture.html', 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Email Capture Example</title>
            <!-- Google Analytics -->
            <script async src="https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX"></script>
            <script>
                window.dataLayer = window.dataLayer || [];
                function gtag(){{dataLayer.push(arguments);}}
                gtag('js', new Date());
                gtag('config', 'G-XXXXXXXXXX');
            </script>
        </head>
        <body style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; padding: 20px;">
            <h1>AuroraScan Security Tool</h1>
            {sample_html}
        </body>
        </html>
        """)

    print("\n✅ ROI Tracking System Ready!")
    print("📊 Dashboard available at: roi_optimization_system.html")
    print("📈 Metrics saved to: roi_data/metrics.json")

if __name__ == '__main__':
    main()