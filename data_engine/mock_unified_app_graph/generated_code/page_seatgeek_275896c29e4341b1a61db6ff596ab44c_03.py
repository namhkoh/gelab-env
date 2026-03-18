# page_id: page_seatgeek_275896c29e4341b1a61db6ff596ab44c_03
# screenshot: 2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6.png
# step_index: 3/9
# task: Open SeatGeek. Look up "Seattle Mariners" tickets. Select the next upcoming event in Los Angeles. Set quantity to 2 and select the best value tickets. Which section are tickets in?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw background, status bar, header/search bar background, separators, and bottom nav background
width, height = canvas.size

# Colors
bg_color = "#ffffff"
status_bar_color = "#efefef"
search_bg = "#f6f6f6"
divider_color = "#e8e8e8"
shadow_color = "#f2f2f2"
nav_divider = "#e9e9e9"

# Fill overall background (canvas starts white but set explicitly)
draw.rectangle([0, 0, width, height], fill=bg_color)

# Status bar area (top)
status_bar_h = 88
draw.rectangle([0, 0, width, status_bar_h], fill=status_bar_color)

# Thin subtle line below status bar
draw.line([(0, status_bar_h), (width, status_bar_h)], fill=divider_color, width=1)

# Search bar card (rounded rectangle)
search_left = 48
search_top = 120
search_right = width - 48
search_h = 144
draw.rounded_rectangle([search_left, search_top, search_right, search_top + search_h],
                       radius=24, fill=search_bg, outline=None)

# Subtle shadow / separator under search card
shadow_top = search_top + search_h
draw.rectangle([search_left + 8, shadow_top, search_right - 8, shadow_top + 6], fill=shadow_color)

# Horizontal divider under search area (full width with side padding)
divider_y1 = search_top + search_h + 24
draw.line([(40, divider_y1), (width - 40, divider_y1)], fill=divider_color, width=1)

# Section divider between lists (approx. where recent searches end / suggestions begin)
# Using approximate coordinate inferred from layout
divider_y2 = 1312
draw.line([(40, divider_y2), (width - 40, divider_y2)], fill=divider_color, width=1)

# Additional faint rule slightly above suggestions header to add depth
draw.line([(40, divider_y2 + 44), (width - 40, divider_y2 + 44)], fill=shadow_color, width=1)

# Bottom navigation bar background and top divider/shadow
nav_h = 168
nav_top = height - nav_h
draw.rectangle([0, nav_top, width, height], fill=bg_color)
draw.line([(0, nav_top), (width, nav_top)], fill=nav_divider, width=2)
draw.line([(0, nav_top + 2), (width, nav_top + 2)], fill=shadow_color, width=1)

# Small top header band (beneath status bar) to visually separate status and content
header_band_h = status_bar_h + 12
draw.rectangle([0, status_bar_h, width, header_band_h], fill=bg_color)

# Optional subtle left/right content gutters (visual guides)
gutter_x = 40
draw.line([(gutter_x, header_band_h), (gutter_x, height - nav_h - 40)], fill=shadow_color, width=1)
draw.line([(width - gutter_x, header_band_h), (width - gutter_x, height - nav_h - 40)], fill=shadow_color, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/00_icon_Bruno_Mars.png
try:
    _c0 = get_crop(0, 1440, 168)
    canvas.paste(_c0, (0, 639), _c0)
except Exception:
    pass
layout["Bruno_Mars"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/01_icon_Madison_Square_Garden.png
try:
    _c1 = get_crop(1, 1440, 168)
    canvas.paste(_c1, (0, 471), _c1)
except Exception:
    pass
layout["Madison_Square_Garden"] = [0, 471, 1440, 639]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/02_icon_7.48_my.png
try:
    _c2 = get_crop(2, 168, 144)
    canvas.paste(_c2, (48, 120), _c2)
except Exception:
    pass
layout["7.48_my"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/03_icon_Tracking.png
try:
    _c3 = get_crop(3, 288, 168)
    canvas.paste(_c3, (864, 2792), _c3)
except Exception:
    pass
layout["Tracking"] = [864, 2792, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/04_icon_Coldplay.png
try:
    _c4 = get_crop(4, 1440, 168)
    canvas.paste(_c4, (0, 1143), _c4)
except Exception:
    pass
layout["Coldplay"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/05_icon_Browse.png
try:
    _c5 = get_crop(5, 288, 168)
    canvas.paste(_c5, (0, 2792), _c5)
except Exception:
    pass
layout["Browse"] = [0, 2792, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/06_icon_L_Olympia_Olympia_Theatre.png
try:
    _c6 = get_crop(6, 1440, 168)
    canvas.paste(_c6, (0, 807), _c6)
except Exception:
    pass
layout["L'Olympia_(Olympia_Theatr"] = [0, 807, 1440, 975]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 48, 69)
    canvas.paste(_c7, (1152, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1152, 0, 1200, 69]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/08_icon_L_Olympia_Olympia_Theatre.png
try:
    _c8 = get_crop(8, 1440, 168)
    canvas.paste(_c8, (0, 639), _c8)
except Exception:
    pass
layout["L'Olympia_(Olympia_Theatr"] = [0, 639, 1440, 807]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/09_icon_icon_9.png
try:
    _c9 = get_crop(9, 61, 64)
    canvas.paste(_c9, (243, 2), _c9)
except Exception:
    pass
layout["icon_9"] = [243, 2, 304, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/10_icon_Tickets.png
try:
    _c10 = get_crop(10, 288, 168)
    canvas.paste(_c10, (576, 2792), _c10)
except Exception:
    pass
layout["Tickets"] = [576, 2792, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/11_icon_Just_Announced_by_My_Performers.png
try:
    _c11 = get_crop(11, 1440, 168)
    canvas.paste(_c11, (0, 1688), _c11)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1688, 1440, 1856]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/12_icon_L_Olympia_Olympia_Theatre.png
try:
    _c12 = get_crop(12, 1440, 168)
    canvas.paste(_c12, (0, 975), _c12)
except Exception:
    pass
layout["L'Olympia_(Olympia_Theatr"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 104, 67)
    canvas.paste(_c13, (1212, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1212, 0, 1316, 67]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/14_icon_Clear.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 120), _c14)
except Exception:
    pass
layout["Clear"] = [1248, 120, 1392, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/15_icon_Coldplay.png
try:
    _c15 = get_crop(15, 1440, 168)
    canvas.paste(_c15, (0, 975), _c15)
except Exception:
    pass
layout["Coldplay"] = [0, 975, 1440, 1143]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/16_icon_Account.png
try:
    _c16 = get_crop(16, 288, 168)
    canvas.paste(_c16, (1152, 2792), _c16)
except Exception:
    pass
layout["Account"] = [1152, 2792, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/17_icon_Events_by_My_Performers.png
try:
    _c17 = get_crop(17, 1440, 168)
    canvas.paste(_c17, (0, 1520), _c17)
except Exception:
    pass
layout["Events_by_My_Performers"] = [0, 1520, 1440, 1688]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 53, 64)
    canvas.paste(_c18, (1319, 1), _c18)
except Exception:
    pass
layout["icon_18"] = [1319, 1, 1372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/19_icon_7.48_my.png
try:
    _c19 = get_crop(19, 47, 63)
    canvas.paste(_c19, (186, 1), _c19)
except Exception:
    pass
layout["7.48_my"] = [186, 1, 233, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 59, 64)
    canvas.paste(_c20, (313, 2), _c20)
except Exception:
    pass
layout["icon_20"] = [313, 2, 372, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/21_icon_Search.png
try:
    _c21 = get_crop(21, 288, 162)
    canvas.paste(_c21, (288, 2792), _c21)
except Exception:
    pass
layout["Search"] = [288, 2792, 576, 2954]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/22_icon_Performer_event_or_venue.png
try:
    _c22 = get_crop(22, 1032, 144)
    canvas.paste(_c22, (216, 120), _c22)
except Exception:
    pass
layout["Performer;_event;_or_venu"] = [216, 120, 1248, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/23_icon_Denver_Nuggets.png
try:
    _c23 = get_crop(23, 1440, 168)
    canvas.paste(_c23, (0, 1143), _c23)
except Exception:
    pass
layout["Denver_Nuggets"] = [0, 1143, 1440, 1311]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/24_icon_Just_Announced_by_My_Performers.png
try:
    _c24 = get_crop(24, 1440, 168)
    canvas.paste(_c24, (0, 1856), _c24)
except Exception:
    pass
layout["Just_Announced_by_My_Perf"] = [0, 1856, 1440, 2024]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/25_text_7.48_my.png
try:
    _c25 = get_crop(25, 153, 52)
    canvas.paste(_c25, (19, 9), _c25)
except Exception:
    pass
layout["7.48_my"] = [19, 9, 172, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/26_text_Recent_searches.png
try:
    _c26 = get_crop(26, 168, 144)
    canvas.paste(_c26, (48, 120), _c26)
except Exception:
    pass
layout["Recent_searches"] = [48, 120, 216, 264]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/seatgeek/275896c29e4341b1a61db6ff596ab44c/step_03_2024_4_22_19_46_275896c29e4341b1a61db6ff596ab44c-6/27_text_Suggestions.png
try:
    _c27 = get_crop(27, 331, 74)
    canvas.paste(_c27, (40, 1423), _c27)
except Exception:
    pass
layout["Suggestions"] = [40, 1423, 371, 1497]
