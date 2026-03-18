# page_id: page_eventbrite_b28077cf24f341ff9de826ac8bd7fb2b_04
# screenshot: 2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6.png
# step_index: 4/16
# task: Open Eventbrite. Explore 'Wellness' events in Washington. Filter to only show free events. Add the first non-promoted event to favorite and follow the organizer.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw UI background and structural elements for the provided canvas.
# Available variables: canvas (PIL Image 1440x2960 RGB), draw (ImageDraw)
# Fonts available but not used here: font_sm, font_md, font_lg, font_xl

W, H = canvas.size

# Overall page background (slightly off-white like the screenshot)
draw.rectangle([0, 0, W, H], fill="#FBFBFD")

# Status bar (top ~56px) - dark gray bar like Android status bar in the screenshot
status_h = 56
draw.rectangle([0, 0, W, status_h], fill="#9E9E9E")

# Header / Search area (white) below status bar
header_top = status_h
header_bottom = 260  # leave room for header + location row
draw.rectangle([0, header_top, W, header_bottom], fill="#FFFFFF")

# Header bottom divider (thin)
draw.line([(24, header_bottom), (W-24, header_bottom)], fill="#E6E6E9", width=2)

# Subtle horizontal rule under filter chips area (chips themselves will be pasted on top)
chips_div_y = 520
draw.line([(24, chips_div_y), (W-24, chips_div_y)], fill="#F0F1F3", width=1)

# Left content margin guide (light vertical subtle line for structure)
draw.line([(48, header_bottom+12), (48, H - 160)], fill="#FFFFFF", width=1)

# Card-like background for the first large event image (rounded rectangle with light shadow)
# Detected primary event image area: pos=(48,676) size=1344x1175
card1_x0, card1_y0 = 48, 676
card1_x1, card1_y1 = card1_x0 + 1344, card1_y0 + 1175

# Shadow (slightly larger, light gray)
shadow_offset = 6
draw.rounded_rectangle(
    [card1_x0 + shadow_offset, card1_y0 + shadow_offset, card1_x1 + shadow_offset, card1_y1 + shadow_offset],
    radius=16, fill="#EDEFF1"
)

# Card background (white)
draw.rounded_rectangle([card1_x0, card1_y0, card1_x1, card1_y1], radius=16, fill="#FFFFFF", outline="#ECEEF0", width=1)

# Thin separator beneath the first card
sep_y1 = card1_y1 + 26
draw.line([(48, sep_y1), (W-48, sep_y1)], fill="#ECEFF2", width=1)

# Second event card area (rounded rectangle)
# Detected second event image area: pos=(48,1899) size=1344x917
card2_x0, card2_y0 = 48, 1899
card2_x1, card2_y1 = card2_x0 + 1344, card2_y0 + 917

# Shadow for second card
draw.rounded_rectangle(
    [card2_x0 + shadow_offset, card2_y0 + shadow_offset, card2_x1 + shadow_offset, card2_y1 + shadow_offset],
    radius=16, fill="#EDEFF1"
)

# Card background
draw.rounded_rectangle([card2_x0, card2_y0, card2_x1, card2_y1], radius=16, fill="#FFFFFF", outline="#ECEEF0", width=1)

# Separator lines between list items (subtle)
draw.line([(48, card2_y1 + 28), (W-48, card2_y1 + 28)], fill="#F1F3F5", width=1)
draw.line([(24, 340), (W-24, 340)], fill="#F3F4F6", width=1)

# Large content area background band behind list (slightly warmer tint under the header to separate from status bar)
list_band_top = header_bottom
list_band_bottom = H - 200
draw.rectangle([0, list_band_top, W, list_band_bottom], fill="#FFFFFF")

# Bottom navigation bar background (keep it clean white with a top divider)
nav_h = 156
nav_top = H - nav_h
draw.rectangle([0, nav_top, W, H], fill="#FFFFFF")
draw.line([(0, nav_top), (W, nav_top)], fill="#E6E7EA", width=2)

# Small top shadow for the bottom nav to lift it slightly
draw.line([(0, nav_top+2), (W, nav_top+2)], fill="#F5F6F8", width=1)

# Accent left guide under "10,000 events" heading area (visual structure only)
heading_region_y = 410
draw.line([(48, heading_region_y - 36), (W-48, heading_region_y - 36)], fill="#FFFFFF", width=1)

# Right-side safe margin line (visual alignment helper)
draw.line([(W-48, header_bottom+12), (W-48, H - 160)], fill="#FFFFFF", width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/00_icon_Music.png
try:
    _c0 = get_crop(0, 187, 103)
    canvas.paste(_c0, (837, 410), _c0)
except Exception:
    pass
layout["Music"] = [837, 410, 1024, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/01_icon_Business.png
try:
    _c1 = get_crop(1, 241, 103)
    canvas.paste(_c1, (1036, 410), _c1)
except Exception:
    pass
layout["Business"] = [1036, 410, 1277, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/02_icon_Anytime.png
try:
    _c2 = get_crop(2, 400, 103)
    canvas.paste(_c2, (425, 410), _c2)
except Exception:
    pass
layout["Anytime"] = [425, 410, 825, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/03_icon_Filters.png
try:
    _c3 = get_crop(3, 359, 103)
    canvas.paste(_c3, (54, 410), _c3)
except Exception:
    pass
layout["Filters"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/04_icon_Foo.png
try:
    _c4 = get_crop(4, 149, 110)
    canvas.paste(_c4, (1282, 406), _c4)
except Exception:
    pass
layout["Foo"] = [1282, 406, 1431, 516]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/05_icon_Flying_or_Falling.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1092, 2415), _c5)
except Exception:
    pass
layout["Flying_or_Falling?"] = [1092, 2415, 1236, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/06_icon_Voilc.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1092, 1192), _c6)
except Exception:
    pass
layout["Voilc^"] = [1092, 1192, 1236, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/07_icon_Flying_or_Falling.png
try:
    _c7 = get_crop(7, 144, 144)
    canvas.paste(_c7, (1236, 2415), _c7)
except Exception:
    pass
layout["Flying_or_Falling?"] = [1236, 2415, 1380, 2559]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/08_icon_Voilc.png
try:
    _c8 = get_crop(8, 144, 144)
    canvas.paste(_c8, (1236, 1192), _c8)
except Exception:
    pass
layout["Voilc^"] = [1236, 1192, 1380, 1336]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/09_icon_Foo.png
try:
    _c9 = get_crop(9, 144, 144)
    canvas.paste(_c9, (1248, 96), _c9)
except Exception:
    pass
layout["Foo"] = [1248, 96, 1392, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/10_icon_4.44.png
try:
    _c10 = get_crop(10, 125, 115)
    canvas.paste(_c10, (54, 113), _c10)
except Exception:
    pass
layout["4.44"] = [54, 113, 179, 228]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/11_icon_Ln_Ccu.png
try:
    _c11 = get_crop(11, 225, 229)
    canvas.paste(_c11, (691, 912), _c11)
except Exception:
    pass
layout["Ln_Ccu"] = [691, 912, 916, 1141]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/12_icon_Wellness.png
try:
    _c12 = get_crop(12, 66, 64)
    canvas.paste(_c12, (308, 0), _c12)
except Exception:
    pass
layout["Wellness"] = [308, 0, 374, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/13_icon_ELOAT_Fen__fade.png
try:
    _c13 = get_crop(13, 1344, 1175)
    canvas.paste(_c13, (48, 676), _c13)
except Exception:
    pass
layout["ELOAT:_Fen__fade"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 51, 62)
    canvas.paste(_c14, (249, 1), _c14)
except Exception:
    pass
layout["icon_14"] = [249, 1, 300, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/15_icon_4.44.png
try:
    _c15 = get_crop(15, 58, 63)
    canvas.paste(_c15, (182, 0), _c15)
except Exception:
    pass
layout["4.44"] = [182, 0, 240, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/16_icon_Online.png
try:
    _c16 = get_crop(16, 377, 144)
    canvas.paste(_c16, (0, 259), _c16)
except Exception:
    pass
layout["Online"] = [0, 259, 377, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/17_icon_4.44.png
try:
    _c17 = get_crop(17, 60, 64)
    canvas.paste(_c17, (114, 0), _c17)
except Exception:
    pass
layout["4.44"] = [114, 0, 174, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/18_icon_icon_18.png
try:
    _c18 = get_crop(18, 103, 61)
    canvas.paste(_c18, (1206, 0), _c18)
except Exception:
    pass
layout["icon_18"] = [1206, 0, 1309, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/19_icon_Wellness.png
try:
    _c19 = get_crop(19, 1344, 191)
    canvas.paste(_c19, (48, 72), _c19)
except Exception:
    pass
layout["Wellness"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/20_icon_icon_20.png
try:
    _c20 = get_crop(20, 59, 61)
    canvas.paste(_c20, (1318, 0), _c20)
except Exception:
    pass
layout["icon_20"] = [1318, 0, 1377, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/21_icon_When_Things_Fall_Apart_Am.png
try:
    _c21 = get_crop(21, 1344, 917)
    canvas.paste(_c21, (48, 1899), _c21)
except Exception:
    pass
layout["When_Things_Fall_Apart:_A"] = [48, 1899, 1392, 2816]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/22_icon_FoUtnn.png
try:
    _c22 = get_crop(22, 241, 296)
    canvas.paste(_c22, (440, 907), _c22)
except Exception:
    pass
layout["(FoUtnn"] = [440, 907, 681, 1203]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/23_icon_Wed_Mav_1_._2_00_PM_EDT.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (288, 2804), _c23)
except Exception:
    pass
layout["Wed,_Mav_1_._2:00_PM_EDT"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/24_icon_Wellness.png
try:
    _c24 = get_crop(24, 48, 61)
    canvas.paste(_c24, (384, 2), _c24)
except Exception:
    pass
layout["Wellness"] = [384, 2, 432, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/25_icon_Flying_or_Falling.png
try:
    _c25 = get_crop(25, 288, 156)
    canvas.paste(_c25, (864, 2804), _c25)
except Exception:
    pass
layout["Flying_or_Falling?"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/26_icon_4.44.png
try:
    _c26 = get_crop(26, 94, 63)
    canvas.paste(_c26, (12, 0), _c26)
except Exception:
    pass
layout["4.44"] = [12, 0, 106, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/27_icon_Promoted.png
try:
    _c27 = get_crop(27, 260, 69)
    canvas.paste(_c27, (70, 1741), _c27)
except Exception:
    pass
layout["Promoted"] = [70, 1741, 330, 1810]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/28_icon_Flying_or_Falling.png
try:
    _c28 = get_crop(28, 288, 156)
    canvas.paste(_c28, (1152, 2804), _c28)
except Exception:
    pass
layout["Flying_or_Falling?"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/29_icon_Wed_Mav_1_._2_00_PM_EDT.png
try:
    _c29 = get_crop(29, 288, 156)
    canvas.paste(_c29, (0, 2804), _c29)
except Exception:
    pass
layout["Wed,_Mav_1_._2:00_PM_EDT"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/30_icon_When_Things_Fall_Apart_Am.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (576, 2804), _c30)
except Exception:
    pass
layout["When_Things_Fall_Apart:_A"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/31_text_10_000_events.png
try:
    _c31 = get_crop(31, 359, 103)
    canvas.paste(_c31, (54, 410), _c31)
except Exception:
    pass
layout["10,000_events"] = [54, 410, 413, 513]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/32_text_Free.png
try:
    _c32 = get_crop(32, 80, 38)
    canvas.paste(_c32, (117, 1391), _c32)
except Exception:
    pass
layout["Free"] = [117, 1391, 197, 1429]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/33_text_Free_Financial_Wellness_Webinar.png
try:
    _c33 = get_crop(33, 1344, 1175)
    canvas.paste(_c33, (48, 676), _c33)
except Exception:
    pass
layout["Free_Financial_Wellness_W"] = [48, 676, 1392, 1851]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/34_text_My.png
try:
    _c34 = get_crop(34, 93, 72)
    canvas.paste(_c34, (1006, 1457), _c34)
except Exception:
    pass
layout["My"] = [1006, 1457, 1099, 1529]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/35_text_Definition_of_Wealth.png
try:
    _c35 = get_crop(35, 540, 66)
    canvas.paste(_c35, (94, 1532), _c35)
except Exception:
    pass
layout["Definition_of_Wealth"] = [94, 1532, 634, 1598]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/36_text_Thu.png
try:
    _c36 = get_crop(36, 91, 45)
    canvas.paste(_c36, (94, 1625), _c36)
except Exception:
    pass
layout["Thu,"] = [94, 1625, 185, 1670]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/37_text_25.png
try:
    _c37 = get_crop(37, 64, 45)
    canvas.paste(_c37, (260, 1620), _c37)
except Exception:
    pass
layout["25"] = [260, 1620, 324, 1665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/38_text_1I_00AM_EDT.png
try:
    _c38 = get_crop(38, 272, 45)
    canvas.paste(_c38, (348, 1620), _c38)
except Exception:
    pass
layout["1I:00AM_EDT"] = [348, 1620, 620, 1665]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/b28077cf24f341ff9de826ac8bd7fb2b/step_04_2024_4_24_16_43_b28077cf24f341ff9de826ac8bd7fb2b-6/39_text_Online.png
try:
    _c39 = get_crop(39, 129, 45)
    canvas.paste(_c39, (91, 1687), _c39)
except Exception:
    pass
layout["Online"] = [91, 1687, 220, 1732]
