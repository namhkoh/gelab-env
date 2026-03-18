# page_id: page_eventbrite_e794243d416840069b0e5f15aefc4a34_02
# screenshot: 2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4.png
# step_index: 2/7
# task: Open Eventbrite. Open "Business Seminar". Select the first event. Note the contact details of the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the provided canvas.
# Available variables: canvas (PIL Image 1440x2960), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Overall background (canvas starts white; reinforce to ensure uniform)
draw.rectangle([(0, 0), (1440, 2960)], fill=(255, 255, 255))

# Status bar area at top (~72px high) - neutral gray
STATUS_H = 72
draw.rectangle([(0, 0), (1440, STATUS_H)], fill=(189, 189, 189))

# Subtle bottom border for status bar
draw.line([(0, STATUS_H), (1440, STATUS_H)], fill=(170, 170, 170), width=1)

# Header / search area background (keeps white, but add subtle shadow)
HEADER_TOP = STATUS_H
HEADER_BOTTOM = 260  # covers the region where the search bar sits; actual text/icons will be pasted on top
draw.rectangle([(0, HEADER_TOP), (1440, HEADER_BOTTOM)], fill=(255, 255, 255))

# Blue underline divider under the search field (approx where the app shows the accent line)
underline_left = 48
underline_right = 1440 - 48
underline_y = HEADER_TOP + 70  # place underline roughly mid-way through header region
underline_thickness = 6
draw.rectangle([(underline_left, underline_y),
                (underline_right, underline_y + underline_thickness)],
               fill=(34, 71, 255))  # vivid blue

# Thin light shadow line immediately under the blue underline for depth
draw.line([(underline_left, underline_y + underline_thickness + 1),
           (underline_right, underline_y + underline_thickness + 1)],
          fill=(220, 220, 225), width=1)

# Content card / section background (rounded rectangle behind the "Recent" list)
# Keep it very subtle (almost white) so pasted text/icons remain primary
card_left = 24
card_right = 1440 - 24
card_top = 240
card_bottom = 1920
card_radius = 12
draw.rounded_rectangle([(card_left, card_top), (card_right, card_bottom)],
                       radius=card_radius, fill=(250, 251, 253), outline=None)

# Separator line under the "Recent" section header (subtle)
# The "Recent" text will be pasted on top; this is a visual separator for the group.
sep_y = 360
draw.line([(card_left + 16, sep_y), (card_right - 16, sep_y)], fill=(235, 236, 240), width=1)

# Additional faint separators to suggest grouped list (not per-item; just section divisions)
# Place a couple to visually separate blocks of items without matching any specific detected element.
draw.line([(card_left + 16, 680), (card_right - 16, 680)], fill=(245, 246, 248), width=1)
draw.line([(card_left + 16, 1100), (card_right - 16, 1100)], fill=(245, 246, 248), width=1)
draw.line([(card_left + 16, 1520), (card_right - 16, 1520)], fill=(245, 246, 248), width=1)

# Bottom navigation bar background and top divider
NAV_TOP = 2804
draw.rectangle([(0, NAV_TOP), (1440, 2960)], fill=(250, 250, 252))
draw.line([(0, NAV_TOP), (1440, NAV_TOP)], fill=(220, 220, 226), width=2)

# Slight inner shadow at the top of bottom nav for depth
draw.line([(0, NAV_TOP + 2), (1440, NAV_TOP + 2)], fill=(235, 235, 238), width=1)

# Subtle left and right page edge dividers (very faint) to frame content area
draw.line([(0, STATUS_H), (0, 2960)], fill=(245, 245, 247), width=1)
draw.line([(1439, STATUS_H), (1439, 2960)], fill=(245, 245, 247), width=1)

# Light vignette at top of content card for separation from header (very subtle)
vignette_top_y = card_top
draw.rectangle([(card_left, vignette_top_y), (card_right, vignette_top_y + 8)], fill=(248, 249, 250))

# End of structural drawing. UI elements (icons/text) will be pasted on top separately.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/00_icon_Cancel.png
try:
    _c0 = get_crop(0, 149, 144)
    canvas.paste(_c0, (1243, 97), _c0)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/01_icon_5.20.png
try:
    _c1 = get_crop(1, 57, 63)
    canvas.paste(_c1, (115, 2), _c1)
except Exception:
    pass
layout["5.20"] = [115, 2, 172, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/02_icon_5.20.png
try:
    _c2 = get_crop(2, 56, 62)
    canvas.paste(_c2, (181, 2), _c2)
except Exception:
    pass
layout["5.20"] = [181, 2, 237, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/03_icon_Search_for__..png
try:
    _c3 = get_crop(3, 60, 62)
    canvas.paste(_c3, (311, 2), _c3)
except Exception:
    pass
layout["[Search_for__."] = [311, 2, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 46, 59)
    canvas.paste(_c4, (251, 4), _c4)
except Exception:
    pass
layout["icon_4"] = [251, 4, 297, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/05_icon_Cancel.png
try:
    _c5 = get_crop(5, 57, 63)
    canvas.paste(_c5, (1316, 0), _c5)
except Exception:
    pass
layout["Cancel"] = [1316, 0, 1373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/06_icon_Cancel.png
try:
    _c6 = get_crop(6, 97, 63)
    canvas.paste(_c6, (1213, 0), _c6)
except Exception:
    pass
layout["Cancel"] = [1213, 0, 1310, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/07_icon_Tickets.png
try:
    _c7 = get_crop(7, 288, 156)
    canvas.paste(_c7, (864, 2804), _c7)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/08_icon_Language_Learning.png
try:
    _c8 = get_crop(8, 1344, 144)
    canvas.paste(_c8, (48, 534), _c8)
except Exception:
    pass
layout["Language_Learning"] = [48, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/09_icon_Close_current_screen.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 1254), _c9)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/10_icon_Close_current_screen.png
try:
    _c10 = get_crop(10, 144, 144)
    canvas.paste(_c10, (1248, 822), _c10)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/11_icon_5.20.png
try:
    _c11 = get_crop(11, 125, 109)
    canvas.paste(_c11, (50, 115), _c11)
except Exception:
    pass
layout["5.20"] = [50, 115, 175, 224]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/12_icon_Photography.png
try:
    _c12 = get_crop(12, 1344, 144)
    canvas.paste(_c12, (48, 1254), _c12)
except Exception:
    pass
layout["Photography"] = [48, 1254, 1392, 1398]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/13_icon_Gardening.png
try:
    _c13 = get_crop(13, 1344, 144)
    canvas.paste(_c13, (48, 678), _c13)
except Exception:
    pass
layout["Gardening"] = [48, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/14_icon_Close_current_screen.png
try:
    _c14 = get_crop(14, 144, 144)
    canvas.paste(_c14, (1248, 1110), _c14)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/15_icon_Close_current_screen.png
try:
    _c15 = get_crop(15, 144, 144)
    canvas.paste(_c15, (1248, 1398), _c15)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/16_icon_Close_current_screen.png
try:
    _c16 = get_crop(16, 144, 144)
    canvas.paste(_c16, (1248, 534), _c16)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 534, 1392, 678]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/17_icon_Close_current_screen.png
try:
    _c17 = get_crop(17, 144, 144)
    canvas.paste(_c17, (1248, 678), _c17)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 678, 1392, 822]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/18_icon_Wellness.png
try:
    _c18 = get_crop(18, 1344, 144)
    canvas.paste(_c18, (48, 1398), _c18)
except Exception:
    pass
layout["Wellness"] = [48, 1398, 1392, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/19_icon_Close_current_screen.png
try:
    _c19 = get_crop(19, 144, 144)
    canvas.paste(_c19, (1248, 1686), _c19)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1686, 1392, 1830]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/20_icon_Close_current_screen.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1248, 1542), _c20)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/21_icon_Favorites.png
try:
    _c21 = get_crop(21, 288, 156)
    canvas.paste(_c21, (576, 2804), _c21)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/22_icon_Close_current_screen.png
try:
    _c22 = get_crop(22, 144, 144)
    canvas.paste(_c22, (1248, 966), _c22)
except Exception:
    pass
layout["Close_current_screen"] = [1248, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/23_icon_Cancel.png
try:
    _c23 = get_crop(23, 144, 144)
    canvas.paste(_c23, (1248, 390), _c23)
except Exception:
    pass
layout["Cancel"] = [1248, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/24_icon_Home.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (0, 2804), _c24)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/25_icon_Search_events.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (288, 2804), _c25)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/26_icon_Cooking.png
try:
    _c26 = get_crop(26, 1344, 144)
    canvas.paste(_c26, (48, 1542), _c26)
except Exception:
    pass
layout["Cooking"] = [48, 1542, 1392, 1686]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/27_icon_Open_Mic_Night.png
try:
    _c27 = get_crop(27, 1344, 144)
    canvas.paste(_c27, (48, 1110), _c27)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 1110, 1392, 1254]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/28_icon_Search_for__..png
try:
    _c28 = get_crop(28, 1344, 191)
    canvas.paste(_c28, (48, 72), _c28)
except Exception:
    pass
layout["[Search_for__."] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/29_icon_Language_Learning.png
try:
    _c29 = get_crop(29, 1344, 144)
    canvas.paste(_c29, (48, 390), _c29)
except Exception:
    pass
layout["Language_Learning"] = [48, 390, 1392, 534]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/30_icon_Sports.png
try:
    _c30 = get_crop(30, 116, 130)
    canvas.paste(_c30, (26, 1697), _c30)
except Exception:
    pass
layout["Sports"] = [26, 1697, 142, 1827]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/31_icon_More.png
try:
    _c31 = get_crop(31, 288, 156)
    canvas.paste(_c31, (1152, 2804), _c31)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/32_icon_Open_Mic_Night.png
try:
    _c32 = get_crop(32, 1344, 144)
    canvas.paste(_c32, (48, 966), _c32)
except Exception:
    pass
layout["Open_Mic_Night"] = [48, 966, 1392, 1110]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/33_icon_Education.png
try:
    _c33 = get_crop(33, 1344, 144)
    canvas.paste(_c33, (48, 822), _c33)
except Exception:
    pass
layout["Education"] = [48, 822, 1392, 966]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/34_text_5.20.png
try:
    _c34 = get_crop(34, 89, 43)
    canvas.paste(_c34, (22, 17), _c34)
except Exception:
    pass
layout["5.20"] = [22, 17, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/35_text_Recent.png
try:
    _c35 = get_crop(35, 200, 56)
    canvas.paste(_c35, (46, 301), _c35)
except Exception:
    pass
layout["Recent"] = [46, 301, 246, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/36_text_Sports.png
try:
    _c36 = get_crop(36, 135, 54)
    canvas.paste(_c36, (160, 1737), _c36)
except Exception:
    pass
layout["Sports"] = [160, 1737, 295, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/e794243d416840069b0e5f15aefc4a34/step_02_2024_4_24_17_19_e794243d416840069b0e5f15aefc4a34-4/37_clickable_Sports.png
try:
    _c37 = get_crop(37, 1344, 144)
    canvas.paste(_c37, (48, 1686), _c37)
except Exception:
    pass
layout["Sports"] = [48, 1686, 1392, 1830]
