# page_id: page_eventbrite_e1a6a0d0e93c4b71830358b28372ec21_02
# screenshot: 2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4.png
# step_index: 2/9
# task: Open Eventbrite. Search for "Language Learning". Filter only online events. Note how many events are available for "Spanish".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the canvas provided.
# Available variables: canvas (PIL Image), draw (PIL ImageDraw), font_sm, font_md, font_lg, font_xl

# Full-canvas subtle off-white background
draw.rectangle([(0, 0), (1440, 2960)], fill=(250, 250, 252))

# Status bar (top area) - slightly darker gray band
STATUS_BAR_H = 72
draw.rectangle([(0, 0), (1440, STATUS_BAR_H)], fill=(189, 189, 189))

# Thin divider under status bar
draw.line([(0, STATUS_BAR_H), (1440, STATUS_BAR_H)], fill=(220, 220, 224), width=1)

# Search/header area underline (primary accent)
# The main search block sits starting around y=72 per layout; draw a prominent blue underline across the content width.
SEARCH_LEFT = 48
SEARCH_RIGHT = 48 + 1344  # content width from layout
# Place the blue underline roughly mid-way through the search header block
blue_underline_y = 72 + 96
draw.line([(SEARCH_LEFT, blue_underline_y), (SEARCH_RIGHT, blue_underline_y)], fill=(32, 85, 255), width=5)

# Light subtle divider below the search/header (secondary thin rule)
draw.line([(SEARCH_LEFT, blue_underline_y + 8), (SEARCH_RIGHT, blue_underline_y + 8)], fill=(235, 235, 240), width=1)

# Separators between list items (use positions derived from the layout)
# Items start near y=390 and are 144px tall; draw lines at the bottoms of each item
item_top_positions = [390, 534, 678, 822, 966, 1110, 1254, 1398, 1542, 1686]
item_height = 144
separator_color = (235, 235, 240)
for y in item_top_positions:
    bottom_y = y + item_height
    # draw across the content width (respect left/right margins)
    draw.line([(SEARCH_LEFT, bottom_y), (SEARCH_RIGHT, bottom_y)], fill=separator_color, width=1)

# Draw a faint vertical guide to emphasize content margin (very subtle, won't conflict with icons/text)
draw.line([(SEARCH_LEFT, STATUS_BAR_H + 10), (SEARCH_LEFT, 2800)], fill=(248, 248, 250), width=1)
draw.line([(SEARCH_RIGHT, STATUS_BAR_H + 10), (SEARCH_RIGHT, 2800)], fill=(248, 248, 250), width=1)

# Bottom navigation bar background and top border
BOTTOM_BAR_TOP = 2804
draw.rectangle([(0, BOTTOM_BAR_TOP), (1440, 2960)], fill=(255, 255, 255))
draw.line([(0, BOTTOM_BAR_TOP), (1440, BOTTOM_BAR_TOP)], fill=(225, 225, 230), width=2)

# Slight shadow/soft divider just above the bottom bar to separate content area from navigation
draw.line([(SEARCH_LEFT, BOTTOM_BAR_TOP - 2), (SEARCH_RIGHT, BOTTOM_BAR_TOP - 2)], fill=(245, 245, 247), width=1)

# Optional: gentle large-area tonal blocks for grouping (very subtle, non-intrusive)
# These are behind groups of rows to help visual grouping but avoid drawing over detected content regions.
group_blocks = [
    (SEARCH_LEFT, 300, SEARCH_RIGHT, 540),   # header zone under "Recent"
    (SEARCH_LEFT, 540, SEARCH_RIGHT, 1440),  # main list zone (keeps subtle tint)
]
for (x1, y1, x2, y2) in group_blocks:
    draw.rectangle([(x1, y1), (x2, y2)], fill=(255, 255, 255))  # keep white to avoid duplicating content

# A faint left column alignment dot grid (only faint alignment guides, very light)
for yy in range(STATUS_BAR_H + 40, BOTTOM_BAR_TOP - 40, 240):
    draw.line([(SEARCH_LEFT, yy), (SEARCH_RIGHT, yy)], fill=(250, 250, 251), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/00_icon_5.17.png
try:
    _c0 = get_crop(0, 61, 64)
    canvas.paste(_c0, (113, 1), _c0)
except Exception:
    pass
layout["5.17"] = [113, 1, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/01_icon_5.17.png
try:
    _c1 = get_crop(1, 58, 63)
    canvas.paste(_c1, (181, 0), _c1)
except Exception:
    pass
layout["5.17"] = [181, 0, 239, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/02_icon_Search_forae.png
try:
    _c2 = get_crop(2, 63, 64)
    canvas.paste(_c2, (309, 1), _c2)
except Exception:
    pass
layout["Search_forae"] = [309, 1, 372, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/03_icon_icon_3.png
try:
    _c3 = get_crop(3, 49, 61)
    canvas.paste(_c3, (249, 2), _c3)
except Exception:
    pass
layout["icon_3"] = [249, 2, 298, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/04_icon_Cancel.png
try:
    _c4 = get_crop(4, 58, 62)
    canvas.paste(_c4, (1316, 0), _c4)
except Exception:
    pass
layout["Cancel"] = [1316, 0, 1374, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 98, 62)
    canvas.paste(_c5, (1212, 0), _c5)
except Exception:
    pass
layout["Cancel"] = [1212, 0, 1310, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/06_icon_Gardening.png
try:
    _c6 = get_crop(6, 1344, 144)
    canvas.paste(_c6, (48, 534), _c6)
except Exception:
    pass
layout["Gardening"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/07_icon_Cancel.png
try:
    _c7 = get_crop(7, 149, 144)
    canvas.paste(_c7, (1243, 97), _c7)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/08_icon_Search_forae.png
try:
    _c8 = get_crop(8, 1344, 144)
    canvas.paste(_c8, (48, 390), _c8)
except Exception:
    pass
layout["Search_forae"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/09_icon_Favorites.png
try:
    _c9 = get_crop(9, 288, 156)
    canvas.paste(_c9, (576, 2804), _c9)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/10_icon_Close_current_screen.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 822), _c10)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/11_icon_Tickets.png
try:
    _c11 = get_crop(11, 288, 156)
    canvas.paste(_c11, (864, 2804), _c11)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/12_icon_5.17.png
try:
    _c12 = get_crop(12, 127, 113)
    canvas.paste(_c12, (53, 115), _c12)
except Exception:
    pass
layout["5.17"] = [53, 115, 180, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/13_icon_Close_current_screen.png
try:
    _c13 = get_crop(13, 144, 144)
    canvas.paste(_c13, (1248, 534), _c13)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 1254), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/15_icon_Cooking.png
try:
    _c15 = get_crop(15, 1344, 144)
    canvas.paste(_c15, (48, 1398), _c15)
except Exception:
    pass
layout["Cooking"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 678), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/17_icon_Open_Mic_Night.png
try:
    _c17 = get_crop(17, 1344, 144)
    canvas.paste(_c17, (48, 678), _c17)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/18_icon_Close_current_screen.png
try:
    _c18 = get_crop(18, 144, 144)
    canvas.paste(_c18, (1248, 1398), _c18)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/19_icon_Close_current_screen.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 1110), _c19)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 1686), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/21_icon_Close_current_screen.png
try:
    _c21 = get_crop(21, 144, 144)
    canvas.paste(_c21, (1248, 1542), _c21)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/22_icon_Cancel.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1248, 390), _c22)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/23_icon_Home.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (0, 2804), _c23)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/24_icon_Search_events.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (288, 2804), _c24)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/25_icon_Search_forae.png
try:
    _c25 = get_crop(25, 48, 64)
    canvas.paste(_c25, (383, 2), _c25)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 431, 66]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/26_icon_Close_current_screen.png
try:
    _c26 = get_crop(26, 144, 144)
    canvas.paste(_c26, (1248, 966), _c26)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/27_icon_Sports.png
try:
    _c27 = get_crop(27, 1344, 144)
    canvas.paste(_c27, (48, 1542), _c27)
except Exception:
    pass
layout["Sports"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/28_icon_Outdoor.png
try:
    _c28 = get_crop(28, 116, 129)
    canvas.paste(_c28, (26, 1697), _c28)
except Exception:
    pass
layout["Outdoor"] = [26, 1697, 142, 1826]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/29_icon_Wellness.png
try:
    _c29 = get_crop(29, 1344, 144)
    canvas.paste(_c29, (48, 1254), _c29)
except Exception:
    pass
layout["Wellness"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/30_icon_More.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (1152, 2804), _c30)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/31_icon_Search_forae.png
try:
    _c31 = get_crop(31, 1344, 191)
    canvas.paste(_c31, (48, 72), _c31)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/32_icon_Open_Mic_Night.png
try:
    _c32 = get_crop(32, 1344, 144)
    canvas.paste(_c32, (48, 822), _c32)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/33_icon_Photography.png
try:
    _c33 = get_crop(33, 1344, 144)
    canvas.paste(_c33, (48, 1110), _c33)
except Exception:
    pass
layout["Photography"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/34_text_5.17.png
try:
    _c34 = get_crop(34, 87, 43)
    canvas.paste(_c34, (22, 17), _c34)
except Exception:
    pass
layout["5.17"] = [22, 17, 109, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/35_text_Recent.png
try:
    _c35 = get_crop(35, 200, 56)
    canvas.paste(_c35, (46, 301), _c35)
except Exception:
    pass
layout["Recent"] = [46, 301, 246, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/36_text_Photography.png
try:
    _c36 = get_crop(36, 251, 60)
    canvas.paste(_c36, (161, 1014), _c36)
except Exception:
    pass
layout["Photography"] = [161, 1014, 412, 1074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/37_text_Outdoor.png
try:
    _c37 = get_crop(37, 164, 45)
    canvas.paste(_c37, (165, 1738), _c37)
except Exception:
    pass
layout["Outdoor"] = [165, 1738, 329, 1783]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/38_clickable_Photography.png
try:
    _c38 = get_crop(38, 1344, 144)
    canvas.paste(_c38, (48, 966), _c38)
except Exception:
    pass
layout["Photography"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_02_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-4/39_clickable_Outdoor.png
try:
    _c39 = get_crop(39, 1344, 144)
    canvas.paste(_c39, (48, 1686), _c39)
except Exception:
    pass
layout["Outdoor"] = [48, 1686, 1392, 1830]
