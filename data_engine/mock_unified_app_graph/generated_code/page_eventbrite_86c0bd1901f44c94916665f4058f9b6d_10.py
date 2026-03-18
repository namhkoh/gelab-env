# page_id: page_eventbrite_86c0bd1901f44c94916665f4058f9b6d_10
# screenshot: 2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12.png
# step_index: 10/11
# task: Open Eventbrite. Set the city to Los Angeles. Select the 'Food & Drink' category. What's the date of the first event that is not promoted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the mobile UI page

# Canvas dimensions assumed 1440x2960 (provided)
W, H = canvas.size

# Colors
status_bar_color = (190, 190, 190)      # light gray status bar
toolbar_bg = (255, 255, 255)           # white toolbar/search area
divider_color = (220, 220, 225)        # subtle divider
card_shadow = (230, 230, 235)          # subtle shadow behind cards
card_bg = (255, 255, 255)              # card background
page_bg = (250, 250, 252)              # overall page slight off-white
bottom_nav_bg = (255, 255, 255)        # bottom nav background

# Fill overall page background (covers initial white)
draw.rectangle([(0, 0), (W, H)], fill=page_bg)

# Status bar area (top ~72px)
status_h = 72
draw.rectangle([(0, 0), (W, status_h)], fill=status_bar_color)

# Header / search area (from status bar bottom to ~260px)
header_top = status_h
header_bottom = 260
draw.rectangle([(0, header_top), (W, header_bottom)], fill=toolbar_bg)

# Subtle bottom divider under header/search area
draw.line([(48, header_bottom), (W - 48, header_bottom)], fill=divider_color, width=2)

# Filters / chips row area separator (thin line above event list)
filters_row_y = 480
draw.line([(48, filters_row_y), (W - 48, filters_row_y)], fill=divider_color, width=1)

# Event card 1 (rounded white card with subtle shadow)
card1_left, card1_right = 48, W - 48
card1_top = 600
# First event image extends inside this card; draw card background and shadow only
card1_bottom = 1768
shadow_offset = 10
corner_radius = 28

# Shadow behind card 1
draw.rounded_rectangle(
    [(card1_left + shadow_offset, card1_top + shadow_offset),
     (card1_right + shadow_offset, card1_bottom + shadow_offset)],
    radius=corner_radius,
    fill=card_shadow
)
# Card 1 background
draw.rounded_rectangle(
    [(card1_left, card1_top), (card1_right, card1_bottom)],
    radius=corner_radius,
    fill=card_bg
)

# Thin separator between image area and card content (subtle)
sep_y = card1_top + 420
draw.line([(card1_left + 24, sep_y), (card1_right - 24, sep_y)], fill=(245,245,247), width=1)

# Event card 2 (rounded white card with subtle shadow)
card2_top = card1_bottom + 24
card2_bottom = 2760
# Shadow behind card 2
draw.rounded_rectangle(
    [(card1_left + shadow_offset, card2_top + shadow_offset),
     (card1_right + shadow_offset, card2_bottom + shadow_offset)],
    radius=corner_radius,
    fill=card_shadow
)
# Card 2 background
draw.rounded_rectangle(
    [(card1_left, card2_top), (card1_right, card2_bottom)],
    radius=corner_radius,
    fill=card_bg
)

# Divider above bottom navigation
bottom_nav_top = 2804
draw.line([(0, bottom_nav_top), (W, bottom_nav_top)], fill=divider_color, width=2)

# Bottom navigation background (sits above page bottom)
draw.rectangle([(0, bottom_nav_top), (W, H)], fill=bottom_nav_bg)

# Add a faint central elevation line on the page (visual guide under header)
draw.line([(48, header_bottom + 24), (W - 48, header_bottom + 24)], fill=(245,245,247), width=1)

# Small section separators within content area to give structure
for y in (card1_bottom + 8, card2_top + 8):
    draw.line([(48, y), (W - 48, y)], fill=(248,248,249), width=1)

# End of structural drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/00_icon_Food_Drink.png
try:
    _c0 = get_crop(0, 312, 135)
    canvas.paste(_c0, (850, 390), _c0)
except Exception:
    pass
layout["Food_&_Drink"] = [850, 390, 1162, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 400, 135)
    canvas.paste(_c1, (438, 390), _c1)
except Exception:
    pass
layout["Anytime"] = [438, 390, 838, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/02_icon_1_Filter.png
try:
    _c2 = get_crop(2, 372, 135)
    canvas.paste(_c2, (54, 390), _c2)
except Exception:
    pass
layout["1_Filter"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/03_icon_Favorite_button.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 2331), _c3)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2331, 1236, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/04_icon_Overflow_menu_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1236, 2331), _c4)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2331, 1380, 2475]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/05_icon_Ado_Ram.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Ado_Ram"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/06_icon_Ado_Ram.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["Ado_Ram"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/07_icon_icon_7.png
try:
    _c7 = get_crop(7, 53, 65)
    canvas.paste(_c7, (1151, 0), _c7)
except Exception:
    pass
layout["icon_7"] = [1151, 0, 1204, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/08_icon_7.14.png
try:
    _c8 = get_crop(8, 122, 112)
    canvas.paste(_c8, (56, 115), _c8)
except Exception:
    pass
layout["7.14"] = [56, 115, 178, 227]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/09_icon_Cirque_Du_Brunch_CINCO_DE_MAYO_Grand.png
try:
    _c9 = get_crop(9, 1344, 1091)
    canvas.paste(_c9, (48, 676), _c9)
except Exception:
    pass
layout["Cirque_Du_Brunch:_CINCO_D"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 84, 62)
    canvas.paste(_c10, (1212, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1212, 0, 1296, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/11_icon_Search_forae.png
try:
    _c11 = get_crop(11, 67, 63)
    canvas.paste(_c11, (308, 0), _c11)
except Exception:
    pass
layout["Search_forae"] = [308, 0, 375, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/12_icon_7.14.png
try:
    _c12 = get_crop(12, 59, 63)
    canvas.paste(_c12, (182, 0), _c12)
except Exception:
    pass
layout["7.14"] = [182, 0, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/13_icon_7.14.png
try:
    _c13 = get_crop(13, 60, 65)
    canvas.paste(_c13, (114, 0), _c13)
except Exception:
    pass
layout["7.14"] = [114, 0, 174, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 54, 63)
    canvas.paste(_c14, (246, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [246, 1, 300, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/15_icon_CIN_Be.png
try:
    _c15 = get_crop(15, 1344, 1091)
    canvas.paste(_c15, (48, 676), _c15)
except Exception:
    pass
layout["CIN_?_Be"] = [48, 676, 1392, 1767]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/16_icon_Search_forae.png
try:
    _c16 = get_crop(16, 1344, 191)
    canvas.paste(_c16, (48, 72), _c16)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 59, 59)
    canvas.paste(_c17, (1317, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1317, 0, 1376, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/18_icon_Promoted.png
try:
    _c18 = get_crop(18, 277, 71)
    canvas.paste(_c18, (53, 1657), _c18)
except Exception:
    pass
layout["Promoted"] = [53, 1657, 330, 1728]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/19_icon_12_00_PM_PDT.png
try:
    _c19 = get_crop(19, 288, 156)
    canvas.paste(_c19, (576, 2804), _c19)
except Exception:
    pass
layout["12:00_PM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/20_icon_Los_Angeles.png
try:
    _c20 = get_crop(20, 492, 144)
    canvas.paste(_c20, (0, 259), _c20)
except Exception:
    pass
layout["Los_Angeles"] = [0, 259, 492, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/21_icon_5010S_La_Brea_Ave.png
try:
    _c21 = get_crop(21, 43, 55)
    canvas.paste(_c21, (283, 2726), _c21)
except Exception:
    pass
layout["5010S_La_Brea_Ave"] = [283, 2726, 326, 2781]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/22_icon_More.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (1152, 2804), _c22)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/23_icon_BE.png
try:
    _c23 = get_crop(23, 1344, 1001)
    canvas.paste(_c23, (48, 1815), _c23)
except Exception:
    pass
layout["BE"] = [48, 1815, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/24_icon_Search_forae.png
try:
    _c24 = get_crop(24, 49, 62)
    canvas.paste(_c24, (384, 2), _c24)
except Exception:
    pass
layout["Search_forae"] = [384, 2, 433, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/25_icon_12_00_PM_PDT.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (864, 2804), _c25)
except Exception:
    pass
layout["12:00_PM_PDT"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/26_icon_7.14.png
try:
    _c26 = get_crop(26, 98, 64)
    canvas.paste(_c26, (12, 0), _c26)
except Exception:
    pass
layout["7.14"] = [12, 0, 110, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/27_icon_5010S_La_Brea_Ave.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (288, 2804), _c27)
except Exception:
    pass
layout["5010S_La_Brea_Ave"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/28_icon_icon_28.png
try:
    _c28 = get_crop(28, 41, 61)
    canvas.paste(_c28, (1273, 0), _c28)
except Exception:
    pass
layout["icon_28"] = [1273, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/29_text_1_712events.png
try:
    _c29 = get_crop(29, 372, 135)
    canvas.paste(_c29, (54, 390), _c29)
except Exception:
    pass
layout["1,712events"] = [54, 390, 426, 525]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/30_text_Sun_May_5.png
try:
    _c30 = get_crop(30, 223, 57)
    canvas.paste(_c30, (90, 1533), _c30)
except Exception:
    pass
layout["Sun,_May_5"] = [90, 1533, 313, 1590]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/31_text_3.00_PM_PDT.png
try:
    _c31 = get_crop(31, 258, 50)
    canvas.paste(_c31, (331, 1533), _c31)
except Exception:
    pass
layout["3.00_PM_PDT"] = [331, 1533, 589, 1583]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/32_text_Station1640.png
try:
    _c32 = get_crop(32, 239, 45)
    canvas.paste(_c32, (94, 1604), _c32)
except Exception:
    pass
layout["Station1640"] = [94, 1604, 333, 1649]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/86c0bd1901f44c94916665f4058f9b6d/step_10_2024_4_23_19_12_86c0bd1901f44c94916665f4058f9b6d-12/33_clickable_Home.png
try:
    _c33 = get_crop(33, 288, 156)
    canvas.paste(_c33, (0, 2804), _c33)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]
