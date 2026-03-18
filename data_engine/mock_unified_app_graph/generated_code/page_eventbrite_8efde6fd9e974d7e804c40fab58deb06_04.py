# page_id: page_eventbrite_8efde6fd9e974d7e804c40fab58deb06_04
# screenshot: 2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6.png
# step_index: 4/8
# task: Open Eventbrite. Search for "Education". Filter only online events. Note how many events are available.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the Eventbrite-like mobile UI.
# Uses provided: canvas (1440x2960 PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

w, h = canvas.size

# Colors
status_bg = (200, 200, 200)        # light grey status bar
divider_color = (216, 216, 216)    # thin divider lines
card_bg = (247, 249, 252)          # very light card background
card_shadow = (226, 228, 231)      # subtle shadow
header_bg = (255, 255, 255)        # header white
app_bg = (255, 255, 255)           # overall white background
nav_border = (230, 230, 230)       # top border of bottom nav

# Clear canvas to main background (white)
draw.rectangle([(0, 0), (w, h)], fill=app_bg)

# Status bar (top area)
status_h = 96
draw.rectangle([(0, 0), (w, status_h)], fill=status_bg)

# Header area (below status)
header_h = 140
header_y0 = status_h
header_y1 = header_y0 + header_h
draw.rectangle([(0, header_y0), (w, header_y1)], fill=header_bg)

# Subtle bottom divider under header
draw.line([(24, header_y1), (w-24, header_y1)], fill=divider_color, width=2)

# Thin separator under filters/title area (approx where chips and "4,290 events" sit)
sep_y = 520
draw.line([(24, sep_y), (w-24, sep_y)], fill=divider_color, width=1)

# Event card 1 background (rounded rect) - positioned behind detected image/text group
# Detected event image/card at (48, 676) size 1344x1115
card1_x, card1_y = 48, 676
card1_w, card1_h = 1344, 1115
pad = 12
card1_bbox = (card1_x - pad, card1_y - pad, card1_x + card1_w + pad, card1_y + card1_h + pad + 120)
# shadow
shadow_offset = 8
draw.rounded_rectangle(
    [card1_bbox[0] + shadow_offset, card1_bbox[1] + shadow_offset, card1_bbox[2] + shadow_offset, card1_bbox[3] + shadow_offset],
    radius=28, fill=card_shadow
)
# card fill
draw.rounded_rectangle(card1_bbox, radius=28, fill=card_bg, outline=None)

# Event card 2 background (rounded rect) - positioned behind second detected image/text group
# Detected second event image/card at (48, 1839) size 1344x977
card2_x, card2_y = 48, 1839
card2_w, card2_h = 1344, 977
card2_bbox = (card2_x - pad, card2_y - pad, card2_x + card2_w + pad, card2_y + card2_h + pad + 120)
# shadow
draw.rounded_rectangle(
    [card2_bbox[0] + shadow_offset, card2_bbox[1] + shadow_offset, card2_bbox[2] + shadow_offset, card2_bbox[3] + shadow_offset],
    radius=28, fill=card_shadow
)
# card fill
draw.rounded_rectangle(card2_bbox, radius=28, fill=card_bg, outline=None)

# Small subtle separators between cards and surrounding content
# Divider above card1 (to separate "4,290 events" area)
draw.line([(36, card1_y - 36), (w-36, card1_y - 36)], fill=divider_color, width=1)
# Divider between card1 and card2
draw.line([(36, card2_y - 36), (w-36, card2_y - 36)], fill=divider_color, width=1)

# Floating large image container outlines (for visual structure) - do not draw content inside
# Top banner outline for the first event image area (behind where the image will be pasted)
banner1_bbox = (card1_x + 8, card1_y + 8, card1_x + card1_w - 8, card1_y + int(card1_h * 0.55))
draw.rounded_rectangle(banner1_bbox, radius=20, outline=divider_color, width=1)

# Top banner outline for the second event image area
banner2_bbox = (card2_x + 8, card2_y + 8, card2_x + card2_w - 8, card2_y + int(card2_h * 0.45))
draw.rounded_rectangle(banner2_bbox, radius=20, outline=divider_color, width=1)

# Bottom navigation bar area
nav_h = 120
nav_y0 = h - nav_h
draw.rectangle([(0, nav_y0), (w, h)], fill=header_bg)
# nav top border
draw.line([(0, nav_y0), (w, nav_y0)], fill=nav_border, width=2)

# Subtle top-left and top-right safe-area rounded decorations (to match screenshot edges)
edge_radius = 12
draw.rounded_rectangle([(-4, header_y1 - 6), (w+4, header_y1 + 6)], radius=edge_radius, fill=None, outline=divider_color, width=1)

# Decorative horizontal guides (very subtle) to suggest grouping (do not draw any text or icons)
guide_color = (240, 241, 243)
for y in (card1_y - 120, card1_y + card1_h + 24, card2_y + card2_h + 24):
    if 0 < y < h:
        draw.line([(36, y), (w-36, y)], fill=guide_color, width=1)

# Final subtle vignette around content area to separate from white background
vignette_line_y = card1_y - 24
draw.line([(24, vignette_line_y), (w-24, vignette_line_y)], fill=(245,245,247), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/04_icon_Foo.png
try:
    _c4 = get_crop(4, 150, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1432, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/05_icon_Favorite_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 1192), _c5)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/06_icon_Overflow_menu_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1192), _c6)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/07_icon_Favorite_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1092, 2355), _c7)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2355, 1236, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 2355), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2355, 1380, 2499]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/09_icon_Foo.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/10_icon_6.59.png
try:
    _c10 = get_crop(10, 121, 115)
    canvas.paste(_c10, (55, 113), _c10)
except Exception:
    pass
layout["6.59"] = [55, 113, 176, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/11_icon_Education.png
try:
    _c11 = get_crop(11, 67, 63)
    canvas.paste(_c11, (308, 0), _c11)
except Exception:
    pass
layout["Education"] = [308, 0, 375, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 104, 61)
    canvas.paste(_c12, (1206, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1206, 0, 1310, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 51, 61)
    canvas.paste(_c13, (249, 1), _c13)
except Exception:
    pass
layout["icon_13"] = [249, 1, 300, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/14_icon_6.59.png
try:
    _c14 = get_crop(14, 59, 63)
    canvas.paste(_c14, (182, 0), _c14)
except Exception:
    pass
layout["6.59"] = [182, 0, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/15_icon_New_York.png
try:
    _c15 = get_crop(15, 434, 144)
    canvas.paste(_c15, (0, 259), _c15)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/16_icon_Education.png
try:
    _c16 = get_crop(16, 1344, 191)
    canvas.paste(_c16, (48, 72), _c16)
except Exception:
    pass
layout["Education"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/17_icon_6.59.png
try:
    _c17 = get_crop(17, 59, 64)
    canvas.paste(_c17, (115, 0), _c17)
except Exception:
    pass
layout["6.59"] = [115, 0, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 60, 61)
    canvas.paste(_c18, (1318, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1318, 0, 1378, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/19_icon_Education.png
try:
    _c19 = get_crop(19, 50, 61)
    canvas.paste(_c19, (384, 2), _c19)
except Exception:
    pass
layout["Education"] = [384, 2, 434, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/20_icon_11.30AM_EDT.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (288, 2804), _c20)
except Exception:
    pass
layout["11.30AM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/21_icon_1I_O0AM_EDT.png
try:
    _c21 = get_crop(21, 1344, 1115)
    canvas.paste(_c21, (48, 676), _c21)
except Exception:
    pass
layout["1I:O0AM_EDT"] = [48, 676, 1392, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/22_icon_Day.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (864, 2804), _c22)
except Exception:
    pass
layout["Day"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/23_icon_Mama_Mingle.png
try:
    _c23 = get_crop(23, 1344, 1115)
    canvas.paste(_c23, (48, 676), _c23)
except Exception:
    pass
layout["Mama_Mingle"] = [48, 676, 1392, 1791]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/24_icon_Day.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (1152, 2804), _c24)
except Exception:
    pass
layout["Day"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/25_icon_2-00_PM.png
try:
    _c25 = get_crop(25, 1344, 977)
    canvas.paste(_c25, (48, 1839), _c25)
except Exception:
    pass
layout["'2-00_PM"] = [48, 1839, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/26_icon_Day.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (576, 2804), _c26)
except Exception:
    pass
layout["Day"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/27_icon_Mama_Mingle.png
try:
    _c27 = get_crop(27, 374, 76)
    canvas.paste(_c27, (90, 1466), _c27)
except Exception:
    pass
layout["Mama_Mingle"] = [90, 1466, 464, 1542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/28_icon_Albee_Sauare.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Albee_Sauare"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/29_text_6.59.png
try:
    _c29 = get_crop(29, 91, 45)
    canvas.paste(_c29, (20, 15), _c29)
except Exception:
    pass
layout["6.59"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/8efde6fd9e974d7e804c40fab58deb06/step_04_2024_4_23_18_50_8efde6fd9e974d7e804c40fab58deb06-6/30_text_4_290_events.png
try:
    _c30 = get_crop(30, 359, 103)
    canvas.paste(_c30, (54, 410), _c30)
except Exception:
    pass
layout["4,290_events"] = [54, 410, 413, 513]
