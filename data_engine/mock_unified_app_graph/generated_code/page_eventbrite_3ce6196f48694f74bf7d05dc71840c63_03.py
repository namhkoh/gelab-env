# page_id: page_eventbrite_3ce6196f48694f74bf7d05dc71840c63_03
# screenshot: 2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5.png
# step_index: 3/9
# task: Open Eventbrite. Search for 'coding workshop'. Sort the results by date. Where is the location of the soonest event that is not promoted?
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Draw background and structural elements for the UI
# Available variables: canvas (PIL Image), draw (ImageDraw), font_sm, font_md, font_lg, font_xl

# Colors
status_bg = "#BDBDBD"        # status bar background
accent_blue = "#2A56D6"      # search underline / accent
divider = "#E6E6E9"          # subtle dividers
card_outline = "#F1F2F4"     # card outline / very light border
card_bg = "#FFFFFF"          # card background (page is mainly white)
page_bg = "#FFFFFF"          # overall page background
nav_top_border = "#E7E7E9"   # top border for bottom nav
subtle_shadow = "#FAFAFB"    # subtle shadow behind cards

w, h = canvas.size

# Fill overall background (canvas already white but set explicitly)
draw.rectangle([0, 0, w, h], fill=page_bg)

# 1) Status bar area at top (~50px tall)
status_h = 56
draw.rectangle([0, 0, w, status_h], fill=status_bg)

# 2) Header / toolbar background area (below status bar)
# keep it white but add an accent underline for the search field
header_top = status_h
header_bottom = 160
draw.rectangle([0, header_top, w, header_bottom], fill=page_bg)

# Blue underline (search field indicator) — inset horizontally similar to content margins
underline_left = 48
underline_right = w - 48
underline_y = 132
underline_thickness = 6
draw.rectangle([underline_left, underline_y, underline_right, underline_y + underline_thickness], fill=accent_blue)

# subtle divider/shadow line under header area
draw.rectangle([0, header_bottom, w, header_bottom + 1], fill=divider)

# 3) Section header divider (above the Events list)
# a light horizontal gap and subtle line to separate header from content
section_divider_y = 340
draw.rectangle([48, section_divider_y, w - 48, section_divider_y + 1], fill=divider)

# 4) List item card backgrounds (rounded rectangles behind each event row)
# Row positions inferred from detected crops: tops at 390, 786, 1182, 1578, 1974 each height ~396
row_tops = [390, 786, 1182, 1578, 1974]
row_height = 396
card_left = 48
card_right = w - 48
card_radius = 12

for top in row_tops:
    top_y = top
    bottom_y = top + row_height
    # subtle shadow strip behind card (very light)
    shadow_rect = [card_left, top_y + 6, card_right, bottom_y + 6]
    draw.rectangle(shadow_rect, fill=subtle_shadow)
    # card background (white) with very light outline to create separation
    draw.rounded_rectangle([card_left, top_y, card_right, bottom_y], radius=card_radius, fill=card_bg, outline=card_outline, width=1)

# 5) Separators between rows (thin lines)
for top in row_tops:
    sep_y = top + row_height + 6  # place a light separator slightly below the card/shadow
    # don't draw beyond content area
    if sep_y < h - 2000:
        draw.line([card_left + 10, sep_y, card_right - 10, sep_y], fill=divider, width=1)

# 6) Additional subtle full-width separators where appropriate
# small line below "Events" header area (approx where Events text sits; do not draw text)
draw.line([48, 360, w - 48, 360], fill=divider, width=1)

# 7) Bottom navigation bar background area (bottom ~156px)
nav_top = 2804
nav_bottom = h
draw.rectangle([0, nav_top, w, nav_bottom], fill=page_bg)
# top border for nav
draw.rectangle([0, nav_top, w, nav_top + 1], fill=nav_top_border)

# 8) Very bottom hairline separator (just above nav) to anchor content area
draw.rectangle([0, nav_top - 1, w, nav_top], fill=divider)

# 9) Left and right content margins (visual guides) - subtle vertical rules (very faint)
# These are purely structural and faint so they won't conflict with pasted icons/text.
draw.line([48, header_bottom + 8, 48, nav_top - 8], fill="#FBFBFC", width=1)
draw.line([w - 48, header_bottom + 8, w - 48, nav_top - 8], fill="#FBFBFC", width=1)

# End of structural/background drawing.

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/00_icon_jumostort.png
try:
    _c0 = get_crop(0, 1344, 396)
    canvas.paste(_c0, (48, 1182), _c0)
except Exception:
    pass
layout["jumostort"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/01_icon_jumostort.png
try:
    _c1 = get_crop(1, 1344, 396)
    canvas.paste(_c1, (48, 786), _c1)
except Exception:
    pass
layout["jumostort"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/02_icon_ronics.png
try:
    _c2 = get_crop(2, 1344, 396)
    canvas.paste(_c2, (48, 1578), _c2)
except Exception:
    pass
layout["ronics"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/03_icon_Coding_Workshop.png
try:
    _c3 = get_crop(3, 1344, 191)
    canvas.paste(_c3, (48, 72), _c3)
except Exception:
    pass
layout["Coding_Workshop]"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/04_icon_icon_4.png
try:
    _c4 = get_crop(4, 50, 57)
    canvas.paste(_c4, (316, 5), _c4)
except Exception:
    pass
layout["icon_4"] = [316, 5, 366, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/05_icon_MCIC_Event_Space_G22.png
try:
    _c5 = get_crop(5, 1344, 396)
    canvas.paste(_c5, (48, 1578), _c5)
except Exception:
    pass
layout["MCIC_Event_Space_(G22)"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/06_icon_icon_6.png
try:
    _c6 = get_crop(6, 40, 54)
    canvas.paste(_c6, (255, 7), _c6)
except Exception:
    pass
layout["icon_6"] = [255, 7, 295, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/07_icon_7.24.png
try:
    _c7 = get_crop(7, 51, 60)
    canvas.paste(_c7, (185, 3), _c7)
except Exception:
    pass
layout["7.24"] = [185, 3, 236, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/08_icon_7.24.png
try:
    _c8 = get_crop(8, 54, 61)
    canvas.paste(_c8, (116, 3), _c8)
except Exception:
    pass
layout["7.24"] = [116, 3, 170, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/09_icon_Tickets.png
try:
    _c9 = get_crop(9, 288, 156)
    canvas.paste(_c9, (864, 2804), _c9)
except Exception:
    pass
layout["Tickets"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/10_icon_7.24.png
try:
    _c10 = get_crop(10, 91, 61)
    canvas.paste(_c10, (16, 2), _c10)
except Exception:
    pass
layout["7.24"] = [16, 2, 107, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/11_icon_Programming_Ne.png
try:
    _c11 = get_crop(11, 76, 233)
    canvas.paste(_c11, (291, 2064), _c11)
except Exception:
    pass
layout["Programming_Ne"] = [291, 2064, 367, 2297]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/12_icon_Cancel.png
try:
    _c12 = get_crop(12, 46, 57)
    canvas.paste(_c12, (1322, 4), _c12)
except Exception:
    pass
layout["Cancel"] = [1322, 4, 1368, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/13_icon_Programming_Ne.png
try:
    _c13 = get_crop(13, 1344, 396)
    canvas.paste(_c13, (48, 1974), _c13)
except Exception:
    pass
layout["Programming_Ne"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/14_icon_7.24.png
try:
    _c14 = get_crop(14, 107, 99)
    canvas.paste(_c14, (62, 121), _c14)
except Exception:
    pass
layout["7.24"] = [62, 121, 169, 220]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/15_icon_Cancel.png
try:
    _c15 = get_crop(15, 66, 61)
    canvas.paste(_c15, (1217, 1), _c15)
except Exception:
    pass
layout["Cancel"] = [1217, 1, 1283, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/16_icon_Cancel.png
try:
    _c16 = get_crop(16, 41, 59)
    canvas.paste(_c16, (1272, 4), _c16)
except Exception:
    pass
layout["Cancel"] = [1272, 4, 1313, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/17_icon_Cancel.png
try:
    _c17 = get_crop(17, 149, 144)
    canvas.paste(_c17, (1243, 97), _c17)
except Exception:
    pass
layout["Cancel"] = [1243, 97, 1392, 241]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/18_icon_Sat.png
try:
    _c18 = get_crop(18, 1344, 396)
    canvas.paste(_c18, (48, 390), _c18)
except Exception:
    pass
layout["Sat,"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/19_icon_Jumpstart_Coding_Workshop.png
try:
    _c19 = get_crop(19, 1344, 396)
    canvas.paste(_c19, (48, 1182), _c19)
except Exception:
    pass
layout["Jumpstart_Coding_Workshop"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/20_icon_Cancel.png
try:
    _c20 = get_crop(20, 144, 144)
    canvas.paste(_c20, (1099, 96), _c20)
except Exception:
    pass
layout["Cancel"] = [1099, 96, 1243, 240]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/21_icon_Jumpstart_Coding_Workshop.png
try:
    _c21 = get_crop(21, 1344, 396)
    canvas.paste(_c21, (48, 1182), _c21)
except Exception:
    pass
layout["Jumpstart_Coding_Workshop"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/22_icon_More.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (1152, 2804), _c22)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/23_icon_Arduino_electronics_and_coding_workshop.png
try:
    _c23 = get_crop(23, 1344, 396)
    canvas.paste(_c23, (48, 1578), _c23)
except Exception:
    pass
layout["Arduino_electronics_and_c"] = [48, 1578, 1392, 1974]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/24_icon_icon_24.png
try:
    _c24 = get_crop(24, 44, 57)
    canvas.paste(_c24, (385, 5), _c24)
except Exception:
    pass
layout["icon_24"] = [385, 5, 429, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/25_icon_Coding_Workshop.png
try:
    _c25 = get_crop(25, 1344, 396)
    canvas.paste(_c25, (48, 390), _c25)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/26_icon_Coding_Workshop.png
try:
    _c26 = get_crop(26, 1344, 396)
    canvas.paste(_c26, (48, 390), _c26)
except Exception:
    pass
layout["Coding_Workshop"] = [48, 390, 1392, 786]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/27_icon_Search_events.png
try:
    _c27 = get_crop(27, 288, 156)
    canvas.paste(_c27, (288, 2804), _c27)
except Exception:
    pass
layout["Search_events"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/28_icon_12.30_PM_EDT.png
try:
    _c28 = get_crop(28, 1344, 396)
    canvas.paste(_c28, (48, 1182), _c28)
except Exception:
    pass
layout["12.30_PM_EDT"] = [48, 1182, 1392, 1578]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/29_icon_Jumpstart_Coding_Workshop.png
try:
    _c29 = get_crop(29, 1344, 396)
    canvas.paste(_c29, (48, 786), _c29)
except Exception:
    pass
layout["Jumpstart_Coding_Workshop"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/30_icon_Home.png
try:
    _c30 = get_crop(30, 288, 156)
    canvas.paste(_c30, (0, 2804), _c30)
except Exception:
    pass
layout["Home"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/31_text_Events.png
try:
    _c31 = get_crop(31, 186, 56)
    canvas.paste(_c31, (46, 301), _c31)
except Exception:
    pass
layout["Events"] = [46, 301, 232, 357]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/32_text_Sat_May_18.png
try:
    _c32 = get_crop(32, 207, 45)
    canvas.paste(_c32, (392, 905), _c32)
except Exception:
    pass
layout["Sat,_May_18"] = [392, 905, 599, 950]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/33_text_4_00_PM_EDT.png
try:
    _c33 = get_crop(33, 1344, 396)
    canvas.paste(_c33, (48, 786), _c33)
except Exception:
    pass
layout["4:00_PM_EDT"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/34_text_Ann_Arbor_District_Library_Pittsfield_Br.png
try:
    _c34 = get_crop(34, 1344, 396)
    canvas.paste(_c34, (48, 786), _c34)
except Exception:
    pass
layout["Ann_Arbor_District_Librar"] = [48, 786, 1392, 1182]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/35_text_Sat.png
try:
    _c35 = get_crop(35, 77, 45)
    canvas.paste(_c35, (390, 2030), _c35)
except Exception:
    pass
layout["Sat,"] = [390, 2030, 467, 2075]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/36_text_4._1O_00_AM_GMT_08.00.png
try:
    _c36 = get_crop(36, 1344, 396)
    canvas.paste(_c36, (48, 1974), _c36)
except Exception:
    pass
layout["4._1O:00_AM_GMT+08.00"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/37_text_Girls_Programming_Network.png
try:
    _c37 = get_crop(37, 1344, 396)
    canvas.paste(_c37, (48, 1974), _c37)
except Exception:
    pass
layout["Girls'_Programming_Networ"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/38_text_Coding.png
try:
    _c38 = get_crop(38, 170, 65)
    canvas.paste(_c38, (1036, 2084), _c38)
except Exception:
    pass
layout["Coding"] = [1036, 2084, 1206, 2149]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/39_text_University_of_Western_Australia_EZONE_St.png
try:
    _c39 = get_crop(39, 1344, 396)
    canvas.paste(_c39, (48, 1974), _c39)
except Exception:
    pass
layout["University_of_Western_Aus"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/40_text_8_93_creator_followers.png
try:
    _c40 = get_crop(40, 1344, 396)
    canvas.paste(_c40, (48, 1974), _c40)
except Exception:
    pass
layout["8_93_creator_followers"] = [48, 1974, 1392, 2370]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/3ce6196f48694f74bf7d05dc71840c63/step_03_2024_4_23_19_21_3ce6196f48694f74bf7d05dc71840c63-5/41_clickable_Favorites.png
try:
    _c41 = get_crop(41, 288, 156)
    canvas.paste(_c41, (576, 2804), _c41)
except Exception:
    pass
layout["Favorites"] = [576, 2804, 864, 2960]
