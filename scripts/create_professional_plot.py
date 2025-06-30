#!/usr/bin/env python3
# fancy svg plot generator

import pandas as pd
import numpy as np
from datetime import datetime
import math

def load_and_prepare_data():
    print("Loading data...")
    
    data = pd.read_csv('results/performance_comparison_data.csv')
    perf_summary = pd.read_csv('results/performance_summary.csv')
    
    data['Date'] = pd.to_datetime(data['Date'])
    
    print(f"Got {len(data)} points")
    
    return data, perf_summary

def create_svg_plot(data, perf_summary):
    # make the svg plot
    
    # dimensions
    width = 1200
    height = 800
    margin_left = 100
    margin_right = 200  # need room for stats
    margin_top = 100
    margin_bottom = 80
    
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    
    # get data
    dates = data['Date']
    cvar_returns = data['CVaR_Index']
    equal_returns = data['Equal_Weight'] 
    cap_returns = data['Cap_Weight']
    
    # scale stuff
    min_date = dates.min()
    max_date = dates.max()
    date_range = (max_date - min_date).days
    
    max_return = max(cvar_returns.max(), equal_returns.max(), cap_returns.max())
    y_max = math.ceil(max_return / 1000) * 1000  # round to 1000s
    y_max = max(y_max, 5500)  # min range
    
    def date_to_x(date):
        days_from_start = (date - min_date).days
        return margin_left + (days_from_start / date_range) * plot_width
    
    def return_to_y(ret):
        return margin_top + plot_height - (ret / y_max) * plot_height
    
    # build svg
    svg_lines = []
    svg_lines.append(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">')
    
    # white bg
    svg_lines.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    
    # grid
    # horizontal lines
    for y_val in range(0, int(y_max) + 1, 1000):
        y_pos = return_to_y(y_val)
        svg_lines.append(f'<line x1="{margin_left}" y1="{y_pos}" x2="{margin_left + plot_width}" y2="{y_pos}" stroke="#e0e0e0" stroke-width="0.5"/>')
        svg_lines.append(f'<text x="{margin_left - 10}" y="{y_pos + 5}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#666">{y_val}%</text>')
    
    # vertical lines (yearly)
    for year in range(2010, 2025):
        year_date = pd.Timestamp(f'{year}-01-01')
        if year_date >= min_date and year_date <= max_date:
            x_pos = date_to_x(year_date)
            svg_lines.append(f'<line x1="{x_pos}" y1="{margin_top}" x2="{x_pos}" y2="{margin_top + plot_height}" stroke="#e0e0e0" stroke-width="0.5"/>')
            svg_lines.append(f'<text x="{x_pos}" y="{margin_top + plot_height + 20}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#666">{year}</text>')
    
    # border
    svg_lines.append(f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#333" stroke-width="1"/>')
    
    # make path data
    def create_path_data(dates, returns):
        path_points = []
        for i, (date, ret) in enumerate(zip(dates, returns)):
            x = date_to_x(date)
            y = return_to_y(ret)
            if i == 0:
                path_points.append(f'M {x:.1f} {y:.1f}')
            else:
                path_points.append(f'L {x:.1f} {y:.1f}')
        return ' '.join(path_points)
    
    # FIXME: paths might be getting clipped or coordinates are off?
    # check viewBox, overflow settings, or coordinate transforms
    
    # cvar line (blue)
    cvar_path = create_path_data(dates, cvar_returns)
    svg_lines.append(f'<path d="{cvar_path}" fill="none" stroke="#1f77b4" stroke-width="2.5"/>')
    
    # equal weight (orange)
    equal_path = create_path_data(dates, equal_returns)
    svg_lines.append(f'<path d="{equal_path}" fill="none" stroke="#ff7f0e" stroke-width="2.0"/>')
    
    # cap weight (green, dashed)
    cap_path = create_path_data(dates, cap_returns)
    svg_lines.append(f'<path d="{cap_path}" fill="none" stroke="#2ca02c" stroke-width="2.0" stroke-dasharray="5,5"/>')
    
    # title
    title_x = width / 2
    svg_lines.append(f'<text x="{title_x}" y="40" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="#333">CVaR Index Performance Comparison (2010-2024)</text>')
    
    # subtitle
    svg_lines.append(f'<text x="{title_x}" y="65" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#666">Cumulative Returns - Quarterly Rebalanced Strategies</text>')
    
    # axis labels
    # y-axis
    y_label_x = 25
    y_label_y = margin_top + plot_height / 2
    svg_lines.append(f'<text x="{y_label_x}" y="{y_label_y}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#333" transform="rotate(-90 {y_label_x} {y_label_y})">Cumulative Return (%)</text>')
    
    # x-axis
    x_label_x = margin_left + plot_width / 2
    x_label_y = height - 25
    svg_lines.append(f'<text x="{x_label_x}" y="{x_label_y}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="#333">Year</text>')
    
    # legend
    legend_x = margin_left + 20
    legend_y = margin_top + 20
    
    # legend bg
    svg_lines.append(f'<rect x="{legend_x - 10}" y="{legend_y - 5}" width="200" height="85" fill="white" fill-opacity="0.9" stroke="#ccc" stroke-width="1" rx="5"/>')
    
    # legend items
    legend_items = [
        ("CVaR Index", "#1f77b4", "solid"),
        ("Equal Weight", "#ff7f0e", "solid"),
        ("Cap Weight (SPY)", "#2ca02c", "dashed")
    ]
    
    for i, (label, color, style) in enumerate(legend_items):
        item_y = legend_y + 15 + i * 20
        
        # line sample
        if style == "solid":
            svg_lines.append(f'<line x1="{legend_x}" y1="{item_y}" x2="{legend_x + 25}" y2="{item_y}" stroke="{color}" stroke-width="2.5"/>')
        else:
            svg_lines.append(f'<line x1="{legend_x}" y1="{item_y}" x2="{legend_x + 25}" y2="{item_y}" stroke="{color}" stroke-width="2" stroke-dasharray="5,5"/>')
        
        # text
        svg_lines.append(f'<text x="{legend_x + 35}" y="{item_y + 4}" font-family="Arial, sans-serif" font-size="12" fill="#333">{label}</text>')
    
    # stats box
    stats_x = margin_left + plot_width + 20
    stats_y = margin_top + 20
    
    # final values (this is kinda hacky but works)
    cvar_final = cvar_returns.iloc[-1]
    equal_final = equal_returns.iloc[-1]
    cap_final = cap_returns.iloc[-1]
    
    # stats bg
    svg_lines.append(f'<rect x="{stats_x - 10}" y="{stats_y - 5}" width="160" height="200" fill="#f8f8f8" stroke="#ccc" stroke-width="1" rx="5"/>')
    
    # title
    svg_lines.append(f'<text x="{stats_x}" y="{stats_y + 15}" font-family="Arial, sans-serif" font-size="13" font-weight="bold" fill="#333">Performance Summary</text>')
    
    # hardcoded stats (yeah i know, but it works)
    stats_data = [
        ("", "Return", "Sharpe"),
        ("CVaR Index", f"{cvar_final:.0f}%", "6.96"),
        ("Equal Weight", f"{equal_final:.0f}%", "6.40"),
        ("Cap Weight", f"{cap_final:.0f}%", "0.65"),
        ("", "", ""),
        ("Max Drawdown:", "", ""),
        ("CVaR Index", "2.03%", ""),
        ("Equal Weight", "2.03%", ""),
        ("Cap Weight", "19.60%", "")
    ]
    
    for i, (strategy, ret, sharpe) in enumerate(stats_data):
        stat_y = stats_y + 35 + i * 16
        if strategy == "":
            if ret == "Return":  # header row
                svg_lines.append(f'<text x="{stats_x}" y="{stat_y}" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="#666">{ret}</text>')
                svg_lines.append(f'<text x="{stats_x + 60}" y="{stat_y}" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="#666">{sharpe}</text>')
            elif ret == "Max Drawdown:":
                svg_lines.append(f'<text x="{stats_x}" y="{stat_y}" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="#666">{ret}</text>')
        else:
            svg_lines.append(f'<text x="{stats_x}" y="{stat_y}" font-family="Arial, sans-serif" font-size="10" fill="#333">{strategy}</text>')
            if ret:
                svg_lines.append(f'<text x="{stats_x + 60}" y="{stat_y}" font-family="Arial, sans-serif" font-size="10" fill="#333">{ret}</text>')
            if sharpe:
                svg_lines.append(f'<text x="{stats_x + 100}" y="{stat_y}" font-family="Arial, sans-serif" font-size="10" fill="#333">{sharpe}</text>')
    
    svg_lines.append('</svg>')
    
    return '\n'.join(svg_lines)

def save_svg_and_convert(svg_content):
    # save svg and try to make png
    
    svg_path = 'results/performance_comparison.svg'
    with open(svg_path, 'w') as f:
        f.write(svg_content)
    print(f"Saved SVG")
    
    # try png conversion
    import subprocess
    import os
    
    png_path = 'results/performance_comparison.png'
    
    # different converters to try
    conversion_methods = [
        ['rsvg-convert', '-f', 'png', '-o', png_path, svg_path],
        ['inkscape', '--export-png', png_path, svg_path],
        ['convert', svg_path, png_path]
    ]
    
    for method in conversion_methods:
        try:
            subprocess.run(method, check=True, capture_output=True)
            print(f"Mad PNG too!")
            return png_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    print(f"Damn can't make PNG - no converter found")
    print("   SVG is fine tho")
    return svg_path

def create_html_viewer(svg_path):
    # make html wrapper for viewing
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>CVaR Index Performance Comparison</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        svg {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
        }}
        .info {{
            margin-top: 20px;
            padding: 15px;
            background-color: #e8f4fd;
            border-radius: 5px;
            text-align: left;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>CVaR Index Performance Visualization</h1>
        <p>Professional-quality performance comparison chart generated from backtesting results</p>
        
        <div style="text-align: center;">
            <object data="performance_comparison.svg" type="image/svg+xml" width="1200" height="800">
                <embed src="performance_comparison.svg" type="image/svg+xml" />
            </object>
        </div>
        
        <div class="info">
            <h3>📊 Chart Features:</h3>
            <ul>
                <li><strong>CVaR Index (Blue):</strong> 28.82% annual return, 6.96 Sharpe ratio</li>
                <li><strong>Equal Weight (Orange):</strong> 29.67% annual return, 6.40 Sharpe ratio</li>
                <li><strong>Cap Weight/SPY (Green, Dashed):</strong> 10.50% annual return, 0.65 Sharpe ratio</li>
            </ul>
            
            <h3>🎯 Key Insights:</h3>
            <ul>
                <li>CVaR optimization delivers exceptional risk-adjusted returns</li>
                <li>Both CVaR and Equal Weight strategies significantly outperform market benchmark</li>
                <li>Risk-controlled approaches achieve superior Sharpe ratios with lower drawdowns</li>
            </ul>
        </div>
        
        <p><small>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
    </div>
</body>
</html>"""
    
    html_path = 'results/performance_comparison.html'
    with open(html_path, 'w') as f:
        f.write(html_content)
    print(f"HTML viewer: {html_path}")
    return html_path

def main():
    # lets gooo
    
    print("Making fancy plot")
    print("=" * 60)
    
    # load data
    data, perf_summary = load_and_prepare_data()
    
    # create svg
    print("Building SVG...")
    svg_content = create_svg_plot(data, perf_summary)
    
    # TODO: Fix SVG rendering bug - plot lines don't show up properly
    # See: https://stackoverflow.com/questions/58961116/svg-path-not-rendering-in-browser
    # Something wrong with the path coordinates or stroke settings?
    
    # save it
    # final_path = save_svg_and_convert(svg_content)
    
    # html viewer
    # html_path = create_html_viewer('performance_comparison.svg')
    
    # for now just save the raw svg content
    with open('results/performance_comparison_raw.svg', 'w') as f:
        f.write(svg_content)
    print("SVG rendering broken - saved raw content to performance_comparison_raw.svg")
    
    print("\n" + "=" * 60)
    print("DONE! (kinda)")
    print("=" * 60)
    print(f"Created:")
    print(f"   • Raw SVG: results/performance_comparison_raw.svg")
    # print(f"   • SVG: results/performance_comparison.svg")
    # print(f"   • HTML: results/performance_comparison.html")
    # if final_path.endswith('.png'):
    #     print(f"   • PNG: results/performance_comparison.png")
    
    print(f"\n Features (when it works):")
    print(f"   • Nice colors")
    print(f"   • Clean design")
    print(f"   • Stats included")
    
    print(f"\n Need to debug the SVG path rendering issue")

if __name__ == "__main__":
    main() 