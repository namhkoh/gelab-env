# page_id: page_eventbrite_f1e087441f9e44d997c2a58b9c8b0258_07
# screenshot: 2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9.png
# step_index: 7/10
# task: Open Eventbrite. Find the 'Arts' category. Select events that are available for this weekend. From the results, open the first item and add it to favorite. Follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# draw page background and structural elements for the Eventbrite mobile UI mockup
# Uses provided variables: canvas (PIL Image) and draw (PIL.ImageDraw), and fonts if needed.

# Colors
BG = "#F6F7F9"           # overall page background (very light)
STATUS_BAR = "#BFBFBF"   # top status bar neutral gray
HEADER_BG = "#FFFFFF"    # header/search background
DIVIDER = "#E6E7EA"      # thin dividers
CARD_BG = "#FFFFFF"      # card background
CARD_SHADOW = "#E3E6EA"  # subtle shadow behind cards
NAV_BG = "#FFFFFF"       # bottom navigation bar background
NAV_DIVIDER = "#E2E4E7"  # nav top divider

# fill full canvas background
draw.rectangle([0, 0, canvas.width, canvas.height], fill=BG)

# status bar (approx 50-56px tall)
status_h = 56
draw.rectangle([0, 0, canvas.width, status_h], fill=STATUS_BAR)

# header/search area background (leave space for detected search elements)
# The detected search area spans from y=72 height ~191, so draw a neutral white header panel behind it.
header_top = status_h
header_bottom = 320  # generous area covering filters and search row
draw.rectangle([0, header_top, canvas.width, header_bottom], fill=HEADER_BG)

# subtle bottom divider under header
draw.line([48, header_bottom, canvas.width - 48, header_bottom], fill=DIVIDER, width=1)

# subtle horizontal rule under the small location row (approx near y = 260)
loc_div_y = 260
draw.line([48, loc_div_y, canvas.width - 48, loc_div_y], fill=DIVIDER, width=1)

# Main content area - card containers
# Card 1 container (rounded card with shadow) - corresponds to big event image/card at y=676
card1_x1, card1_y1 = 48, 676
card1_w, card1_h = 1344, 1048
card1_x2, card1_y2 = card1_x1 + card1_w, card1_y1 + card1_h
r = 28  # corner radius

# shadow (slightly offset)
shadow_offset = 8
draw.rounded_rectangle(
    [card1_x1 + shadow_offset, card1_y1 + shadow_offset, card1_x2 + shadow_offset, card1_y2 + shadow_offset],
    radius=r,
    fill=CARD_SHADOW
)

# card background
draw.rounded_rectangle([card1_x1, card1_y1, card1_x2, card1_y2], radius=r, fill=CARD_BG)

# separator line below first card content area (subtle)
sep_y1 = card1_y2 + 28
draw.line([48, sep_y1, canvas.width - 48, sep_y1], fill=DIVIDER, width=1)

# Card 2 container (second event card)
card2_x1, card2_y1 = 48, 1772
card2_w, card2_h = 1344, 1044
card2_x2, card2_y2 = card2_x1 + card2_w, card2_y1 + card2_h

draw.rounded_rectangle(
    [card2_x1 + shadow_offset, card2_y1 + shadow_offset, card2_x2 + shadow_offset, card2_y2 + shadow_offset],
    radius=r,
    fill=CARD_SHADOW
)
draw.rounded_rectangle([card2_x1, card2_y1, card2_x2, card2_y2], radius=r, fill=CARD_BG)

# small divider between list items further down the page
draw.line([48, card2_y2 + 24, canvas.width - 48, card2_y2 + 24], fill=DIVIDER, width=1)

# Draw subtle section separators for the list area (light horizontal rules)
# one above the cards (near events count area)
draw.line([48, 520, canvas.width - 48, 520], fill=DIVIDER, width=1)

# bottom navigation bar background (approx 140px tall)
nav_h = 140
nav_top = canvas.height - nav_h
draw.rectangle([0, nav_top, canvas.width, canvas.height], fill=NAV_BG)
# top divider for nav
draw.line([0, nav_top, canvas.width, nav_top], fill=NAV_DIVIDER, width=1)

# also add faint left/right margins shadow lines to frame content area
draw.line([48, header_top + 8, 48, canvas.height - nav_h - 8], fill=DIVIDER, width=1)
draw.line([canvas.width - 48, header_top + 8, canvas.width - 48, canvas.height - nav_h - 8], fill=DIVIDER, width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/00_icon_This_Weekend.png
try:
    _c0 = get_crop(0, 504, 135)
    canvas.paste(_c0, (458, 390), _c0)
except Exception:
    pass
layout["This_Weekend"] = [458, 390, 962, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/01_icon_2_Filters.png
try:
    _c1 = get_crop(1, 392, 135)
    canvas.paste(_c1, (54, 390), _c1)
except Exception:
    pass
layout["2_Filters"] = [54, 390, 446, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/02_icon_Arts.png
try:
    _c2 = get_crop(2, 152, 135)
    canvas.paste(_c2, (974, 390), _c2)
except Exception:
    pass
layout["Arts"] = [974, 390, 1126, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/03_icon_SHIPYARD.png
try:
    _c3 = get_crop(3, 1344, 1048)
    canvas.paste(_c3, (48, 676), _c3)
except Exception:
    pass
layout["SHIPYARD"] = [48, 676, 1392, 1724]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/05_icon_AprIl_26-27.21.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2288), _c5)
except Exception:
    pass
layout["AprIl_26-27.21"] = [1092, 2288, 1236, 2432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/07_icon_AprIl_26-27.21.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 2288), _c7)
except Exception:
    pass
layout["AprIl_26-27.21"] = [1236, 2288, 1380, 2432]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/08_icon_4.32.png
try:
    _c8 = get_crop(8, 117, 112)
    canvas.paste(_c8, (59, 115), _c8)
except Exception:
    pass
layout["4.32"] = [59, 115, 176, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/09_icon_4.32.png
try:
    _c9 = get_crop(9, 60, 64)
    canvas.paste(_c9, (180, 0), _c9)
except Exception:
    pass
layout["4.32"] = [180, 0, 240, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/10_icon_Search_forae.png
try:
    _c10 = get_crop(10, 65, 62)
    canvas.paste(_c10, (308, 1), _c10)
except Exception:
    pass
layout["Search_forae"] = [308, 1, 373, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/11_icon_1I_00_AM_PDT.png
try:
    _c11 = get_crop(11, 1344, 1048)
    canvas.paste(_c11, (48, 676), _c11)
except Exception:
    pass
layout["1I:00_AM_PDT"] = [48, 676, 1392, 1724]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/12_icon_4.32.png
try:
    _c12 = get_crop(12, 59, 64)
    canvas.paste(_c12, (115, 0), _c12)
except Exception:
    pass
layout["4.32"] = [115, 0, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 53, 62)
    canvas.paste(_c13, (247, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [247, 1, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 57, 61)
    canvas.paste(_c14, (1317, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1317, 0, 1374, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/15_icon_CatchLight_Visual_Storytelling_Summit_20.png
try:
    _c15 = get_crop(15, 288, 156)
    canvas.paste(_c15, (576, 2804), _c15)
except Exception:
    pass
layout["CatchLight_Visual_Storyte"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/16_icon_icon_16.png
try:
    _c16 = get_crop(16, 61, 61)
    canvas.paste(_c16, (1212, 0), _c16)
except Exception:
    pass
layout["icon_16"] = [1212, 0, 1273, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 40, 62)
    canvas.paste(_c17, (1273, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1273, 0, 1313, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/18_icon_Search_forae.png
try:
    _c18 = get_crop(18, 1344, 191)
    canvas.paste(_c18, (48, 72), _c18)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/19_icon_Search_forae.png
try:
    _c19 = get_crop(19, 49, 61)
    canvas.paste(_c19, (383, 2), _c19)
except Exception:
    pass
layout["Search_forae"] = [383, 2, 432, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/20_icon_I_00_PM_PDT.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (288, 2804), _c20)
except Exception:
    pass
layout["I:00_PM_PDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/21_icon_San_Francisco.png
try:
    _c21 = get_crop(21, 536, 144)
    canvas.paste(_c21, (0, 259), _c21)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/22_icon_storytelling.png
try:
    _c22 = get_crop(22, 1344, 1044)
    canvas.paste(_c22, (48, 1772), _c22)
except Exception:
    pass
layout["storytelling"] = [48, 1772, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/23_icon_4.32.png
try:
    _c23 = get_crop(23, 140, 62)
    canvas.paste(_c23, (12, 1), _c23)
except Exception:
    pass
layout["4.32"] = [12, 1, 152, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/24_icon_AprIl_26-27.21.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["AprIl_26-27.21"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/25_icon_CatchLight_Visual_Storytelling_Summit_20.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (864, 2804), _c25)
except Exception:
    pass
layout["CatchLight_Visual_Storyte"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/26_icon_KQED.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["KQED"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/27_text_302_events.png
try:
    _c27 = get_crop(27, 392, 135)
    canvas.paste(_c27, (54, 390), _c27)
except Exception:
    pass
layout["302_events"] = [54, 390, 446, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/28_text_Few_tickets_left.png
try:
    _c28 = get_crop(28, 294, 49)
    canvas.paste(_c28, (130, 2484), _c28)
except Exception:
    pass
layout["Few_tickets_left"] = [130, 2484, 424, 2533]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/29_text_CatchLight_Visual_Storytelling_Summit_20.png
try:
    _c29 = get_crop(29, 1344, 1044)
    canvas.paste(_c29, (48, 1772), _c29)
except Exception:
    pass
layout["CatchLight_Visual_Storyte"] = [48, 1772, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/30_text_Sat.png
try:
    _c30 = get_crop(30, 89, 52)
    canvas.paste(_c30, (90, 2656), _c30)
except Exception:
    pass
layout["Sat,"] = [90, 2656, 179, 2708]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/31_text_27.png
try:
    _c31 = get_crop(31, 62, 43)
    canvas.paste(_c31, (253, 2658), _c31)
except Exception:
    pass
layout["27"] = [253, 2658, 315, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/32_text_I_00_PM_PDT.png
try:
    _c32 = get_crop(32, 249, 45)
    canvas.paste(_c32, (339, 2656), _c32)
except Exception:
    pass
layout["I:00_PM_PDT"] = [339, 2656, 588, 2701]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f1e087441f9e44d997c2a58b9c8b0258/step_07_2024_4_24_16_31_f1e087441f9e44d997c2a58b9c8b0258-9/33_text_KQED.png
try:
    _c33 = get_crop(33, 117, 52)
    canvas.paste(_c33, (90, 2722), _c33)
except Exception:
    pass
layout["KQED"] = [90, 2722, 207, 2774]
