# page_id: page_eventbrite_f502e886c78146dfb2f1efc2a331c781_10
# screenshot: 2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12.png
# step_index: 10/18
# task: Open Eventbrite. Search for 'music festival' in San Francisco. Set date available from April 30 to May 3. How many events are listed?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Paint overall background
bg_color = "#F6F7F9"       # off-white background like the screenshot
status_color = "#BFBFBF"   # gray status bar
divider_color = "#E6E6E6"  # subtle dividers
card_shadow = "#E9E9EB"    # light shadow for cards
card_fill = "#FFFFFF"      # card background (white)
nav_bg = "#FFFFFF"         # bottom navigation background

W, H = canvas.size

# Fill main background
draw.rectangle([(0, 0), (W, H)], fill=bg_color)

# STATUS BAR
status_h = 96
draw.rectangle([(0, 0), (W, status_h)], fill=status_color)

# HEADER / TOOLBAR AREA (keeps background consistent, with bottom divider)
header_top = status_h
header_bottom = 220
draw.rectangle([(0, header_top), (W, header_bottom)], fill=bg_color)
# subtle bottom divider under header
draw.line([(48, header_bottom), (W-48, header_bottom)], fill=divider_color, width=2)

# Light subtle underline beneath filters area (filters are auto-pasted; we only separate sections)
filters_bottom = 520
draw.line([(48, filters_bottom), (W-48, filters_bottom)], fill=divider_color, width=1)

# FIRST EVENT CARD (rounded white card with a faint shadow)
card1_x, card1_y = 48, 676
card1_w, card1_h = 1344, 1175
shadow_offset = 8
radius = 28

# shadow
draw.rounded_rectangle(
    [
        (card1_x + shadow_offset, card1_y + shadow_offset),
        (card1_x + card1_w + shadow_offset, card1_y + card1_h + shadow_offset)
    ],
    radius=radius,
    fill=card_shadow
)
# card background
draw.rounded_rectangle(
    [(card1_x, card1_y), (card1_x + card1_w, card1_y + card1_h)],
    radius=radius,
    fill=card_fill,
    outline=divider_color
)

# subtle separator under first card area (to separate details area from next card)
sep_y = card1_y + card1_h + 20
draw.line([(48, sep_y), (W-48, sep_y)], fill=divider_color, width=1)

# SECOND EVENT CARD (rounded white card with a faint shadow)
card2_x, card2_y = 48, 1899
card2_w, card2_h = 1344, 917

# shadow
draw.rounded_rectangle(
    [
        (card2_x + shadow_offset, card2_y + shadow_offset),
        (card2_x + card2_w + shadow_offset, card2_y + card2_h + shadow_offset)
    ],
    radius=24,
    fill=card_shadow
)
# card background
draw.rounded_rectangle(
    [(card2_x, card2_y), (card2_x + card2_w, card2_y + card2_h)],
    radius=24,
    fill=card_fill,
    outline=divider_color
)

# thin divider above bottom navigation
nav_top = 2804
draw.line([(0, nav_top), (W, nav_top)], fill=divider_color, width=2)

# BOTTOM NAVIGATION BACKGROUND
draw.rectangle([(0, nav_top), (W, H)], fill=nav_bg)

# Final faint full-width separators (visual rhythm between event list items)
# Separator between header and first card's title area (approx)
draw.line([(48, 600), (W-48, 600)], fill=divider_color, width=1)

# small crease/shadow under cards to ground them (very subtle)
draw.line([(card1_x + 10, card1_y + card1_h + 2), (card1_x + card1_w - 10, card1_y + card1_h + 2)], fill="#F1F1F1", width=2)
draw.line([(card2_x + 10, card2_y + card2_h + 2), (card2_x + card2_w - 10, card2_y + card2_h + 2)], fill="#F1F1F1", width=2)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/04_icon_Favorite_button.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1192), _c4)
except Exception:
    pass
layout["Favorite_button"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/05_icon_Foo.png
try:
    _c5 = get_crop(5, 134, 110)
    canvas.paste(_c5, (1284, 406), _c5)
except Exception:
    pass
layout["Foo"] = [1284, 406, 1418, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/06_icon_Favorite_button.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 2415), _c6)
except Exception:
    pass
layout["Favorite_button"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/07_icon_Overflow_menu_button.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 2415), _c7)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/08_icon_Overflow_menu_button.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/09_icon_Foo.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 53, 64)
    canvas.paste(_c10, (1151, 1), _c10)
except Exception:
    pass
layout["icon_10"] = [1151, 1, 1204, 65]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/11_icon_7.19.png
try:
    _c11 = get_crop(11, 130, 120)
    canvas.paste(_c11, (52, 111), _c11)
except Exception:
    pass
layout["7.19"] = [52, 111, 182, 231]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 67, 62)
    canvas.paste(_c12, (307, 1), _c12)
except Exception:
    pass
layout["icon_12"] = [307, 1, 374, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/13_icon_icon_13.png
try:
    _c13 = get_crop(13, 100, 63)
    canvas.paste(_c13, (1212, 0), _c13)
except Exception:
    pass
layout["icon_13"] = [1212, 0, 1312, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/14_icon_Juneteenth_Decades_Festival_Fillmorel_Mu.png
try:
    _c14 = get_crop(14, 1344, 1175)
    canvas.paste(_c14, (48, 676), _c14)
except Exception:
    pass
layout["Juneteenth_Decades_Festiv"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/15_icon_icon_15.png
try:
    _c15 = get_crop(15, 51, 61)
    canvas.paste(_c15, (249, 1), _c15)
except Exception:
    pass
layout["icon_15"] = [249, 1, 300, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/16_icon_7.19.png
try:
    _c16 = get_crop(16, 60, 63)
    canvas.paste(_c16, (181, 0), _c16)
except Exception:
    pass
layout["7.19"] = [181, 0, 241, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/17_icon_icon_17.png
try:
    _c17 = get_crop(17, 57, 61)
    canvas.paste(_c17, (1318, 0), _c17)
except Exception:
    pass
layout["icon_17"] = [1318, 0, 1375, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/18_icon_7.19.png
try:
    _c18 = get_crop(18, 60, 64)
    canvas.paste(_c18, (115, 0), _c18)
except Exception:
    pass
layout["7.19"] = [115, 0, 175, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/19_icon_San_Francisco.png
try:
    _c19 = get_crop(19, 536, 144)
    canvas.paste(_c19, (0, 259), _c19)
except Exception:
    pass
layout["San_Francisco"] = [0, 259, 536, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/20_icon_24_-_Sun_May_26.png
try:
    _c20 = get_crop(20, 288, 156)
    canvas.paste(_c20, (288, 2804), _c20)
except Exception:
    pass
layout["24_-_Sun,_May_26"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/21_icon_Music_Festival.png
try:
    _c21 = get_crop(21, 49, 62)
    canvas.paste(_c21, (384, 2), _c21)
except Exception:
    pass
layout["Music_Festival"] = [384, 2, 433, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/22_icon_Music_Festival.png
try:
    _c22 = get_crop(22, 1344, 191)
    canvas.paste(_c22, (48, 72), _c22)
except Exception:
    pass
layout["Music_Festival"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/23_icon_To_bottlerock_2024.png
try:
    _c23 = get_crop(23, 1344, 917)
    canvas.paste(_c23, (48, 1899), _c23)
except Exception:
    pass
layout["To_bottlerock_2024"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/24_icon_Promoted.png
try:
    _c24 = get_crop(24, 240, 67)
    canvas.paste(_c24, (86, 1743), _c24)
except Exception:
    pass
layout["Promoted"] = [86, 1743, 326, 1810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/25_icon_10.30AM_PDT.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (864, 2804), _c25)
except Exception:
    pass
layout["10.30AM_PDT"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/26_icon_10.30AM_PDT.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (576, 2804), _c26)
except Exception:
    pass
layout["10.30AM_PDT"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/27_icon_Juneteenth_Decades_Festival_Fillmorel_Mu.png
try:
    _c27 = get_crop(27, 1344, 1175)
    canvas.paste(_c27, (48, 676), _c27)
except Exception:
    pass
layout["Juneteenth_Decades_Festiv"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/28_icon_Fri.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (0, 2804), _c28)
except Exception:
    pass
layout["Fri,"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/29_icon_BOTTLEROCK_Preferred_Shuttle_Bus_From_SA.png
try:
    _c29 = get_crop(29, 1344, 917)
    canvas.paste(_c29, (48, 1899), _c29)
except Exception:
    pass
layout["BOTTLEROCK_Preferred_Shut"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/30_icon_More.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (1152, 2804), _c30)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/31_text_7.19.png
try:
    _c31 = get_crop(31, 91, 45)
    canvas.paste(_c31, (20, 15), _c31)
except Exception:
    pass
layout["7.19"] = [20, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/32_text_5_476_events.png
try:
    _c32 = get_crop(32, 359, 103)
    canvas.paste(_c32, (54, 410), _c32)
except Exception:
    pass
layout["5,476_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/33_text_JINIETEEN.png
try:
    _c33 = get_crop(33, 400, 103)
    canvas.paste(_c33, (425, 410), _c33)
except Exception:
    pass
layout["JINIETEEN"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/34_text_JUNE_15.png
try:
    _c34 = get_crop(34, 237, 79)
    canvas.paste(_c34, (1131, 692), _c34)
except Exception:
    pass
layout["JUNE_15"] = [1131, 692, 1368, 771]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/35_text_1330_Fillmore_St.png
try:
    _c35 = get_crop(35, 316, 45)
    canvas.paste(_c35, (91, 1687), _c35)
except Exception:
    pass
layout["1330_Fillmore_St"] = [91, 1687, 407, 1732]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/36_text_LM.png
try:
    _c36 = get_crop(36, 62, 31)
    canvas.paste(_c36, (187, 2019), _c36)
except Exception:
    pass
layout["LM"] = [187, 2019, 249, 2050]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/37_text_BoTTLERock.png
try:
    _c37 = get_crop(37, 314, 63)
    canvas.paste(_c37, (273, 2017), _c37)
except Exception:
    pass
layout["BoTTLERock"] = [273, 2017, 587, 2080]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/f502e886c78146dfb2f1efc2a331c781/step_10_2024_4_23_19_15_f502e886c78146dfb2f1efc2a331c781-12/38_text_Lot.png
try:
    _c38 = get_crop(38, 86, 41)
    canvas.paste(_c38, (1161, 1971), _c38)
except Exception:
    pass
layout["Lot"] = [1161, 1971, 1247, 2012]
