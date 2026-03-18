# page_id: page_eventbrite_1c30518736b1454cb330b963c1cc6d86_02
# screenshot: 2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4.png
# step_index: 2/9
# task: Open Eventbrite. Search for "Open Mic Nights". Filter the results to only include free events. Select the first non-promoted event in the list - what"s the location of that event?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw overall background (ensure clean white)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar (top area)
status_h = 56
status_color = (200, 200, 200)  # light gray status bar
draw.rectangle([(0, 0), (1440, status_h)], fill=status_color)

# Header / search toolbar background area (below status bar)
header_top = status_h
header_bottom = 140
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))

# Thin blue underline for the search field
underline_y = header_bottom
underline_color = (29, 61, 216)  # vivid blue accent
draw.line([(48, underline_y), (1392, underline_y)], fill=underline_color, width=4)

# Subtle divider below the blue line for separation
draw.line([(48, underline_y + 4), (1392, underline_y + 4)], fill=(235, 235, 240), width=1)

# Section separator above the list ("Recent" header area)
recent_divider_y = 300
draw.line([(48, recent_divider_y), (1392, recent_divider_y)], fill=(242, 242, 246), width=1)

# Row separators for the list items (align with detected item widths: x from 48 to 1392)
row_ends = [678, 822, 966, 1110, 1254, 1398, 1542, 1686, 1830]
sep_color = (242, 242, 246)
for y in row_ends:
    # draw a very subtle thin line across the content column
    draw.line([(48, y), (1392, y)], fill=sep_color, width=1)

# Light left and right inset guides (very subtle to suggest content column, not visible as text/icons)
inset_color = (250, 250, 250)
draw.rectangle([(0, recent_divider_y + 2), (48, 2800)], fill=inset_color)
draw.rectangle([(1392, recent_divider_y + 2), (1440, 2800)], fill=inset_color)

# Bottom navigation area separation and background
nav_top = 2804
draw.line([(0, nav_top), (1440, nav_top)], fill=(230, 230, 235), width=2)
# subtle shadow strip above the nav
draw.rectangle([(0, nav_top - 6), (1440, nav_top)], fill=(245, 245, 247))
# nav background (kept white so icons pasted on top are visible)
draw.rectangle([(0, nav_top), (1440, 2960)], fill=(255, 255, 255))

# Optional faint vertical separators for the nav icon columns (so layout is visually guided but icons will be pasted)
nav_cols_x = [288, 576, 864, 1152]
for x in nav_cols_x:
    draw.line([(x, nav_top + 20), (x, 2950)], fill=(255, 255, 255), width=1)

# Gentle overall vignette near top to mimic native app chrome (very subtle)
draw.rectangle([(0, 0), (1440, 8)], fill=(245, 245, 245))

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/00_icon_4.53.png
try:
    _c0 = get_crop(0, 60, 63)
    canvas.paste(_c0, (114, 1), _c0)
except Exception:
    pass
layout["4.53"] = [114, 1, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/01_icon_4.53.png
try:
    _c1 = get_crop(1, 58, 62)
    canvas.paste(_c1, (181, 1), _c1)
except Exception:
    pass
layout["4.53"] = [181, 1, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/02_icon_Search_forae.png
try:
    _c2 = get_crop(2, 63, 62)
    canvas.paste(_c2, (309, 2), _c2)
except Exception:
    pass
layout["Search_forae"] = [309, 2, 372, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 49, 61)
    canvas.paste(_c3, (249, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [249, 2, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/04_icon_Photography.png
try:
    _c4 = get_crop(4, 1344, 144)
    canvas.paste(_c4, (48, 534), _c4)
except Exception:
    pass
layout["Photography"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 57, 62)
    canvas.paste(_c5, (1316, 0), _c5)
except Exception:
    pass
layout["Cancel"] = [1316, 0, 1373, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/06_icon_community_events.png
try:
    _c6 = get_crop(6, 1344, 144)
    canvas.paste(_c6, (48, 1398), _c6)
except Exception:
    pass
layout["community_events"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/07_icon_Cancel.png
try:
    _c7 = get_crop(7, 97, 62)
    canvas.paste(_c7, (1212, 0), _c7)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1309, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/08_icon_Cancel.png
try:
    _c8 = get_crop(8, 149, 144)
    canvas.paste(_c8, (1243, 97), _c8)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/09_icon_community_events.png
try:
    _c9 = get_crop(9, 1344, 144)
    canvas.paste(_c9, (48, 1254), _c9)
except Exception:
    pass
layout["community_events"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/10_icon_4.53.png
try:
    _c10 = get_crop(10, 125, 112)
    canvas.paste(_c10, (54, 115), _c10)
except Exception:
    pass
layout["4.53"] = [54, 115, 179, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/11_icon_community_events.png
try:
    _c11 = get_crop(11, 1344, 144)
    canvas.paste(_c11, (48, 1542), _c11)
except Exception:
    pass
layout["community_events"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/12_icon_Search_forae.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 390), _c12)
except Exception:
    pass
layout["Search_forae"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/13_icon_Tickets.png
try:
    _c13 = get_crop(13, 288, 156)
    canvas.paste(_c13, (864, 2804), _c13)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/14_icon_Wellness.png
try:
    _c14 = get_crop(14, 1344, 144)
    canvas.paste(_c14, (48, 678), _c14)
except Exception:
    pass
layout["Wellness"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/15_icon_Favorites.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (576, 2804), _c15)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 822), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/17_icon_community_events.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 1110), _c17)
except Exception:
    pass
layout["community_events"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/18_icon_Close_current_screen.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 678), _c18)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/19_icon_Art.png
try:
    _c19 = get_crop(19, 115, 130)
    canvas.paste(_c19, (25, 1697), _c19)
except Exception:
    pass
layout["Art"] = [25, 1697, 140, 1827]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 1254), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/21_icon_Close_current_screen.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1248, 1686), _c21)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/22_icon_Close_current_screen.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1248, 534), _c22)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/23_icon_Close_current_screen.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1248, 1398), _c23)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/24_icon_Close_current_screen.png
try:
    _c24 = get_crop(24, 144, 144)
    canvas.paste(_c24, (1248, 1110), _c24)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/25_icon_Search_forae.png
try:
    _c25 = get_crop(25, 48, 64)
    canvas.paste(_c25, (383, 2), _c25)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 431, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/26_icon_Close_current_screen.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (1248, 1542), _c26)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/27_icon_Home.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (0, 2804), _c27)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/28_icon_Cancel.png
try:
    _c28 = get_crop(28, 144, 144)
    canvas.paste(_c28, (1248, 390), _c28)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/29_icon_Search_forae.png
try:
    _c29 = get_crop(29, 1344, 191)
    canvas.paste(_c29, (48, 72), _c29)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/30_icon_Search_events.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (288, 2804), _c30)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/31_icon_Close_current_screen.png
try:
    _c31 = get_crop(31, 144, 144)
    canvas.paste(_c31, (1248, 966), _c31)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/32_icon_4.53.png
try:
    _c32 = get_crop(32, 94, 61)
    canvas.paste(_c32, (13, 2), _c32)
except Exception:
    pass
layout["4.53"] = [13, 2, 107, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/33_icon_More.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (1152, 2804), _c33)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/34_icon_community_events.png
try:
    _c34 = get_crop(34, 1344, 144)
    canvas.paste(_c34, (48, 966), _c34)
except Exception:
    pass
layout["community_events"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/35_icon_Cooking.png
try:
    _c35 = get_crop(35, 1344, 144)
    canvas.paste(_c35, (48, 822), _c35)
except Exception:
    pass
layout["Cooking"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/36_text_Recent.png
try:
    _c36 = get_crop(36, 200, 56)
    canvas.paste(_c36, (46, 301), _c36)
except Exception:
    pass
layout["Recent"] = [46, 301, 246, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/37_text_Art.png
try:
    _c37 = get_crop(37, 67, 45)
    canvas.paste(_c37, (164, 1739), _c37)
except Exception:
    pass
layout["Art"] = [164, 1739, 231, 1784]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/1c30518736b1454cb330b963c1cc6d86/step_02_2024_4_24_16_52_1c30518736b1454cb330b963c1cc6d86-4/38_clickable_Art.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 1686), _c38)
except Exception:
    pass
layout["Art"] = [48, 1686, 1392, 1830]
