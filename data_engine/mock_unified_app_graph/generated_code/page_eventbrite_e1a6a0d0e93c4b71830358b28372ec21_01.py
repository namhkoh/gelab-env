# page_id: page_eventbrite_e1a6a0d0e93c4b71830358b28372ec21_01
# screenshot: 2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3.png
# step_index: 1/9
# task: Open Eventbrite. Search for "Language Learning". Filter only online events. Note how many events are available for "Spanish".
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw general background
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Top status bar (approx 50px high) - muted gray background (no icons/text)
status_bar_h = 50
draw.rectangle([(0, 0), (1440, status_bar_h)], fill=(200, 200, 200))
# subtle divider under status bar
draw.line([(0, status_bar_h), (1440, status_bar_h)], fill=(220, 220, 220), width=1)

# Header / toolbar area (leave search bar area unmodified; just draw subtle bottom divider)
header_top = status_bar_h
header_bottom = 230  # header region height (visual structural area only)
# keep header same white as page but add faint shadow/divider at bottom
draw.rectangle([(0, header_top), (1440, header_bottom)], fill=(255, 255, 255))
draw.line([(24, header_bottom), (1416, header_bottom)], fill=(235, 235, 235), width=1)

# Content section: draw subtle card/background groups for event rows
# Use the detected vertical anchors from the image analysis (structural grouping)
event_row_anchors = [490, 761, 1282, 1678, 2074, 2470]
card_x0 = 48
card_x1 = 48 + 1344  # match detected content width
card_w = card_x1 - card_x0

# Visual card settings
card_fill = (250, 250, 251)         # very subtle off-white for card backgrounds
card_outline = (235, 235, 238)      # light outline to separate from page
card_radius = 16

for y in event_row_anchors:
    # Draw a rounded card background behind each event group.
    # Keep a modest vertical padding to avoid overlapping header/search area.
    y0 = y - 20
    y1 = y + 180
    # Ensure we stay within canvas bounds
    y0 = max(header_bottom + 8, y0)
    y1 = min(2960 - 180, y1)
    draw.rounded_rectangle([(card_x0, y0), (card_x1, y1)],
                           radius=card_radius, fill=card_fill, outline=card_outline, width=1)
    # subtle drop shadow line below each card (very faint)
    shadow_y = y1 + 2
    draw.line([(card_x0 + 8, shadow_y), (card_x1 - 8, shadow_y)], fill=(245, 245, 246), width=2)

# Separator lines between logical sections (not overlapping icon/text areas)
separator_color = (235, 235, 238)
# compute separators roughly between the card anchors
for i in range(len(event_row_anchors) - 1):
    sep_y = (event_row_anchors[i] + event_row_anchors[i+1]) // 2 + 80
    # Clamp within content area
    if header_bottom + 40 < sep_y < 2960 - 220:
        draw.line([(card_x0, sep_y), (card_x1, sep_y)], fill=separator_color, width=1)

# Bottom navigation bar background and top divider (structural only; icons will be pasted on top)
nav_h = 156
nav_top = 2960 - nav_h
draw.rectangle([(0, nav_top), (1440, 2960)], fill=(255, 255, 255))
# top border for nav
draw.line([(0, nav_top), (1440, nav_top)], fill=(230, 230, 233), width=1)

# Small accent: rounded background band behind the central content area near bottom (subtle)
# (This is structural background only, not an icon or text)
band_y0 = 2520
band_y1 = 2620
draw.rounded_rectangle([(160, band_y0), (1280, band_y1)],
                       radius=28, fill=(255, 255, 255), outline=(240, 240, 243), width=1)

# End of UI structural/background drawing.
# (Icons, text and interactive elements will be pasted on top by the caller.)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/00_icon_Chicago.png
try:
    _c0 = get_crop(0, 388, 117)
    canvas.paste(_c0, (526, 2651), _c0)
except Exception:
    pass
layout["Chicago"] = [526, 2651, 914, 2768]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/01_icon_CyPo6.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 1678), _c1)
except Exception:
    pass
layout["CyPo6"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/02_icon_ripg_-_LeaTG_Atans.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 2074), _c2)
except Exception:
    pass
layout["ripg_-_LeaTG_Atans"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/03_icon_Okstore.png
try:
    _c3 = get_crop(3, 1344, 396)
    canvas.paste(_c3, (48, 1282), _c3)
except Exception:
    pass
layout["Okstore"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/04_icon_Search_events.png
try:
    _c4 = get_crop(4, 1179, 144)
    canvas.paste(_c4, (195, 93), _c4)
except Exception:
    pass
layout["Search_events"] = [195, 93, 1374, 237]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/05_icon_Sat_Oct_19.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 490), _c5)
except Exception:
    pass
layout["Sat,_Oct_19"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/06_icon_Dovetail_Brewery.png
try:
    _c6 = get_crop(6, 144, 139)
    canvas.paste(_c6, (1140, 1935), _c6)
except Exception:
    pass
layout["Dovetail_Brewery"] = [1140, 1935, 1284, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 123)
    canvas.paste(_c7, (1140, 2347), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1140, 2347, 1284, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/08_icon_Favorite_button.png
try:
    _c8 = get_crop(8, 144, 139)
    canvas.paste(_c8, (1140, 1539), _c8)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1539, 1284, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/09_icon_Overflow_menu_button.png
try:
    _c9 = get_crop(9, 144, 139)
    canvas.paste(_c9, (1284, 1935), _c9)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1935, 1428, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/10_icon_Overflow_menu_button.png
try:
    _c10 = get_crop(10, 144, 123)
    canvas.paste(_c10, (1284, 2347), _c10)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 2347, 1428, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/11_icon_Favorite_button.png
try:
    _c11 = get_crop(11, 144, 139)
    canvas.paste(_c11, (1140, 1143), _c11)
except Exception:
    pass
layout["Favorite_button"] = [1140, 1143, 1284, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/12_icon_Favorite_button.png
try:
    _c12 = get_crop(12, 144, 125)
    canvas.paste(_c12, (1140, 761), _c12)
except Exception:
    pass
layout["Favorite_button"] = [1140, 761, 1284, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/13_icon_7940_Wolcott_Ave_apt_2_Chicago_IL_USA.png
try:
    _c13 = get_crop(13, 1344, 396)
    canvas.paste(_c13, (48, 490), _c13)
except Exception:
    pass
layout["7940_$_Wolcott_Ave_apt_2,"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/14_icon_Overflow_menu_button.png
try:
    _c14 = get_crop(14, 144, 125)
    canvas.paste(_c14, (1284, 761), _c14)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 761, 1428, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/15_icon_Joliet.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (288, 2804), _c15)
except Exception:
    pass
layout["Joliet"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/16_icon_Overflow_menu_button.png
try:
    _c16 = get_crop(16, 144, 139)
    canvas.paste(_c16, (1284, 1539), _c16)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1539, 1428, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/17_icon_5.17.png
try:
    _c17 = get_crop(17, 105, 100)
    canvas.paste(_c17, (40, 122), _c17)
except Exception:
    pass
layout["5.17"] = [40, 122, 145, 222]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/18_icon_through_thc_chi.png
try:
    _c18 = get_crop(18, 288, 156)
    canvas.paste(_c18, (0, 2804), _c18)
except Exception:
    pass
layout["through_thc_chi"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/19_icon_5.17.png
try:
    _c19 = get_crop(19, 55, 60)
    canvas.paste(_c19, (183, 2), _c19)
except Exception:
    pass
layout["5.17"] = [183, 2, 238, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/20_icon_ON.png
try:
    _c20 = get_crop(20, 1344, 396)
    canvas.paste(_c20, (48, 886), _c20)
except Exception:
    pass
layout["ON"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/21_icon_Overflow_menu_button.png
try:
    _c21 = get_crop(21, 144, 139)
    canvas.paste(_c21, (1284, 1143), _c21)
except Exception:
    pass
layout["Overflow_menu_button"] = [1284, 1143, 1428, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/22_icon_icon_22.png
try:
    _c22 = get_crop(22, 60, 58)
    canvas.paste(_c22, (312, 3), _c22)
except Exception:
    pass
layout["icon_22"] = [312, 3, 372, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/23_icon_49_creator_followers.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 886), _c23)
except Exception:
    pass
layout["49_creator_followers"] = [48, 886, 1392, 1282]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 50, 59)
    canvas.paste(_c24, (248, 2), _c24)
except Exception:
    pass
layout["icon_24"] = [248, 2, 298, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/25_icon_icon_25.png
try:
    _c25 = get_crop(25, 48, 53)
    canvas.paste(_c25, (1321, 7), _c25)
except Exception:
    pass
layout["icon_25"] = [1321, 7, 1369, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/26_icon_Indie_Bookstore_Day_at_Goblin_Market.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 1282), _c26)
except Exception:
    pass
layout["Indie_Bookstore_Day_at_Go"] = [48, 1282, 1392, 1678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/27_icon_Planting_Seeds_bilingual.png
try:
    _c27 = get_crop(27, 1344, 396)
    canvas.paste(_c27, (48, 2074), _c27)
except Exception:
    pass
layout["Planting_Seeds_(bilingual"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/28_icon_5.17.png
try:
    _c28 = get_crop(28, 57, 60)
    canvas.paste(_c28, (115, 3), _c28)
except Exception:
    pass
layout["5.17"] = [115, 3, 172, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/29_icon_icon_29.png
try:
    _c29 = get_crop(29, 58, 58)
    canvas.paste(_c29, (1212, 4), _c29)
except Exception:
    pass
layout["icon_29"] = [1212, 4, 1270, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/30_icon_icon_30.png
try:
    _c30 = get_crop(30, 41, 55)
    canvas.paste(_c30, (1272, 6), _c30)
except Exception:
    pass
layout["icon_30"] = [1272, 6, 1313, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/31_icon_icon_31.png
try:
    _c31 = get_crop(31, 44, 56)
    canvas.paste(_c31, (385, 6), _c31)
except Exception:
    pass
layout["icon_31"] = [385, 6, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/32_icon_73_creator_followers.png
try:
    _c32 = get_crop(32, 1344, 396)
    canvas.paste(_c32, (48, 1678), _c32)
except Exception:
    pass
layout["73_creator_followers"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/33_icon_Self-Love_in_Nature_Releasing_Grief_thro.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 2074), _c33)
except Exception:
    pass
layout["Self-Love_in_Nature:_Rele"] = [48, 2074, 1392, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/34_icon_Grief_R.png
try:
    _c34 = get_crop(34, 1344, 346)
    canvas.paste(_c34, (48, 2470), _c34)
except Exception:
    pass
layout["Grief_R"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/35_icon_6_00_PM_CDT.png
try:
    _c35 = get_crop(35, 1344, 396)
    canvas.paste(_c35, (48, 1678), _c35)
except Exception:
    pass
layout["6:00_PM_CDT"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/36_icon_Dovetail_Brewery.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 1678), _c36)
except Exception:
    pass
layout["Dovetail_Brewery"] = [48, 1678, 1392, 2074]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/37_icon_Discover_Your_Path_To_Healing_With_Our_G.png
try:
    _c37 = get_crop(37, 1344, 346)
    canvas.paste(_c37, (48, 2470), _c37)
except Exception:
    pass
layout["Discover_Your_Path_To_Hea"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/38_text_5.17.png
try:
    _c38 = get_crop(38, 89, 43)
    canvas.paste(_c38, (22, 17), _c38)
except Exception:
    pass
layout["5.17"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/39_text_More_events_you_II_love.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 490), _c39)
except Exception:
    pass
layout["More_events_you'II_love"] = [48, 490, 1392, 886]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/40_text_Tue_May_7.png
try:
    _c40 = get_crop(40, 191, 43)
    canvas.paste(_c40, (390, 2525), _c40)
except Exception:
    pass
layout["Tue,_May_7"] = [390, 2525, 581, 2568]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/41_text_6_00_PM_CDT.png
try:
    _c41 = get_crop(41, 1344, 346)
    canvas.paste(_c41, (48, 2470), _c41)
except Exception:
    pass
layout["6:00_PM_CDT"] = [48, 2470, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/42_text_Joliet.png
try:
    _c42 = get_crop(42, 96, 38)
    canvas.paste(_c42, (390, 2723), _c42)
except Exception:
    pass
layout["Joliet"] = [390, 2723, 486, 2761]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/43_clickable_Favorites.png
try:
    _c43 = get_crop(43, 288, 156)
    canvas.paste(_c43, (576, 2804), _c43)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/44_clickable_Tickets.png
try:
    _c44 = get_crop(44, 288, 156)
    canvas.paste(_c44, (864, 2804), _c44)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e1a6a0d0e93c4b71830358b28372ec21/step_01_2024_4_24_17_16_e1a6a0d0e93c4b71830358b28372ec21-3/45_clickable_More.png
try:
    _c45 = get_crop(45, 288, 156)
    canvas.paste(_c45, (1152, 2804), _c45)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]
