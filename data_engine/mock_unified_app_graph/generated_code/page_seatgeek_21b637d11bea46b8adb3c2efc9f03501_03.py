# page_id: page_seatgeek_21b637d11bea46b8adb3c2efc9f03501_03
# screenshot: 2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6.png
# step_index: 3/10
# task: Open SeatGeek and find the soonest upcoming NBA game in New York with "Nets", record the cheapest price in google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural UI elements for the provided canvas (1440x2960)
# Available variables: canvas (PIL Image), draw (PIL ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
bg_white = "#ffffff"
status_bar_bg = "#f3f3f3"       # light grey status bar
search_bg = "#fafafa"           # search bar background
search_stroke = "#e7e7e7"       # search bar border
divider = "#e8e8e8"             # section dividers
card_bg = "#ffffff"             # subtle card (keeps white)
nav_top_border = "#efefef"      # top border for bottom nav
shadow_line = "#f0f0f0"

# Clear canvas background to dominant color (white)
draw.rectangle([(0, 0), (w, h)], fill=bg_white)

# Status bar area at top (~72px tall)
status_h = 72
draw.rectangle([(0, 0), (w, status_h)], fill=status_bar_bg)

# Subtle top hairline under status bar
draw.line([(0, status_h-1), (w, status_h-1)], fill=shadow_line, width=1)

# Search bar (rounded rect) - spans across with left/right margins
search_left = 48
search_top = 120
search_right = w - 48
search_bottom = search_top + 144
search_radius = 32
draw.rounded_rectangle(
    [(search_left, search_top), (search_right, search_bottom)],
    radius=search_radius,
    fill=search_bg,
    outline=search_stroke,
    width=1
)

# Subtle divider line below search area (full-width inset)
divider_y1 = search_bottom + 36
draw.line([(48, divider_y1), (w-48, divider_y1)], fill=divider, width=1)

# Section grouping: Recent searches area - draw a faint background band to separate visually
# Do not draw any text or icons inside this area; just a subtle panel background behind list region
recent_top = divider_y1 + 24
recent_bottom = 1320  # approximate end of recent searches list
# very light fill (almost white) to indicate grouping
draw.rectangle([(24, recent_top), (w-24, recent_bottom)], fill=card_bg)

# Thin divider at end of recent searches before suggestions
divider_y2 = recent_bottom + 12
draw.line([(48, divider_y2), (w-48, divider_y2)], fill=divider, width=1)

# Suggestions area: leave whitespace but draw a faint rounded card background around suggestions items
suggestions_top = divider_y2 + 36
suggestions_bottom = suggestions_top + 360
card_left = 36
card_right = w - 36
card_radius = 16
# Keep fill white but add a very light border to indicate card boundary
draw.rounded_rectangle(
    [(card_left, suggestions_top), (card_right, suggestions_bottom)],
    radius=card_radius,
    fill=card_bg,
    outline="#fbfbfb",
    width=1
)

# Horizontal separators for potential sections (subtle)
sep_x1 = 48
sep_x2 = w - 48
# a faint separator between suggestion items (three items expected)
sep_y_a = suggestions_top + 82
sep_y_b = suggestions_top + 164
draw.line([(sep_x1, sep_y_a), (sep_x2, sep_y_a)], fill="#fbfbfb", width=1)
draw.line([(sep_x1, sep_y_b), (sep_x2, sep_y_b)], fill="#fbfbfb", width=1)

# Bottom navigation background and top border/shadow
nav_top = 2792
nav_bottom = h
draw.rectangle([(0, nav_top), (w, nav_bottom)], fill=bg_white)
# subtle top border shadow
draw.line([(0, nav_top), (w, nav_top)], fill=nav_top_border, width=2)
# faint gradient-like lines to mimic elevation (stacked faint lines)
draw.line([(0, nav_top+2), (w, nav_top+2)], fill="#f7f7f7", width=1)
draw.line([(0, nav_top+4), (w, nav_top+4)], fill="#fbfbfb", width=1)

# Final micro accents: very light vertical padding guide lines (invisible UI guides)
# (Extremely faint to avoid interfering with pasted content)
guide_color = "#ffffff"
draw.line([(24, 0), (24, h)], fill=guide_color, width=1)
draw.line([(w-24, 0), (w-24, h)], fill=guide_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/00_icon_icon_0.png
try:
    _c0 = get_crop(0, 47, 70)
    canvas.paste(_c0, (1153, 0), _c0)
except Exception:
    pass
layout["icon_0"] = [1153, 0, 1200, 70]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/01_icon_Just_Announced_by_My_Performers.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 1688), _c1)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/02_icon_Tracking.png
try:
    _c2 = get_crop(2, 288, 168)
    canvas.paste(_c2, (864, 2792), _c2)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/03_icon_NBA_Playoffs.png
try:
    _c3 = get_crop(3, 1440, 168)
    canvas.paste(_c3, (0, 807), _c3)
except Exception:
    pass
layout["NBA_Playoffs"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/04_icon_Browse.png
try:
    _c4 = get_crop(4, 288, 168)
    canvas.paste(_c4, (0, 2792), _c4)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/05_icon_6.37.png
try:
    _c5 = get_crop(5, 168, 144)
    canvas.paste(_c5, (48, 120), _c5)
except Exception:
    pass
layout["6.37"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/06_icon_Sofia_Isella.png
try:
    _c6 = get_crop(6, 1440, 168)
    canvas.paste(_c6, (0, 975), _c6)
except Exception:
    pass
layout["Sofia_Isella"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 96, 68)
    canvas.paste(_c7, (1216, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1216, 0, 1312, 68]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/08_icon_Tickets.png
try:
    _c8 = get_crop(8, 288, 168)
    canvas.paste(_c8, (576, 2792), _c8)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/09_icon_Events_by_My_Performers.png
try:
    _c9 = get_crop(9, 1440, 168)
    canvas.paste(_c9, (0, 1520), _c9)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/10_icon_Brooklyn_Nets.png
try:
    _c10 = get_crop(10, 1440, 168)
    canvas.paste(_c10, (0, 471), _c10)
except Exception:
    pass
layout["Brooklyn_Nets"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/11_icon_Clear.png
try:
    _c11 = get_crop(11, 144, 144)
    canvas.paste(_c11, (1248, 120), _c11)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/12_icon_NBA_Playoffs.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 639), _c12)
except Exception:
    pass
layout["NBA_Playoffs"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 52, 67)
    canvas.paste(_c13, (1319, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1319, 0, 1371, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/14_icon_Account.png
try:
    _c14 = get_crop(14, 288, 168)
    canvas.paste(_c14, (1152, 2792), _c14)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/15_icon_Drake.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 639), _c15)
except Exception:
    pass
layout["Drake"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/16_icon_Just_Announced_by_My_Performers.png
try:
    _c16 = get_crop(16, 1440, 168)
    canvas.paste(_c16, (0, 1856), _c16)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/17_icon_Search.png
try:
    _c17 = get_crop(17, 288, 162)
    canvas.paste(_c17, (288, 2792), _c17)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/18_icon_Austin_FC.png
try:
    _c18 = get_crop(18, 1440, 168)
    canvas.paste(_c18, (0, 1143), _c18)
except Exception:
    pass
layout["Austin_FC"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/19_icon_Recent_searches.png
try:
    _c19 = get_crop(19, 1440, 168)
    canvas.paste(_c19, (0, 471), _c19)
except Exception:
    pass
layout["Recent_searches"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/20_icon_Search.png
try:
    _c20 = get_crop(20, 288, 162)
    canvas.paste(_c20, (288, 2792), _c20)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/21_icon_Performer_event_or_venue.png
try:
    _c21 = get_crop(21, 1032, 144)
    canvas.paste(_c21, (216, 120), _c21)
except Exception:
    pass
layout["Performer;_event;_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/22_text_6.37.png
try:
    _c22 = get_crop(22, 89, 45)
    canvas.paste(_c22, (20, 15), _c22)
except Exception:
    pass
layout["6.37"] = [20, 15, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/23_text_EK.png
try:
    _c23 = get_crop(23, 50, 39)
    canvas.paste(_c23, (253, 24), _c23)
except Exception:
    pass
layout["EK"] = [253, 24, 303, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/24_text_Recent_searches.png
try:
    _c24 = get_crop(24, 168, 144)
    canvas.paste(_c24, (48, 120), _c24)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/21b637d11bea46b8adb3c2efc9f03501/step_03_2024_3_20_14_36_21b637d11bea46b8adb3c2efc9f03501-6/25_text_Suggestions.png
try:
    _c25 = get_crop(25, 331, 74)
    canvas.paste(_c25, (40, 1423), _c25)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
