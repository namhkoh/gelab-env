# page_id: page_eventbrite_97cc452004d64d67bb4c5b51ffb8f6d4_06
# screenshot: 2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8.png
# step_index: 6/7
# task: Open Eventbrite. Search Business event. Select the first one that is not promoted. Share to google keep notes.
# current_instruction: 
# previous_instruction: 
# next_instruction: 
# styling_source: gpt
# render_status: render_from_code
# This code targets the original screenshot resolution.
# The final runtime image is then rendered into the 1080x2400 canvas with a top nav strip.

# --- GPT styling skeleton ---
# Background
draw.rectangle((0, 0, 1440, 2960), fill=(250, 250, 252))  # subtle off-white background

# Status bar (top ~72px)
status_h = 72
draw.rectangle((0, 0, 1440, status_h), fill=(217, 217, 217))  # light gray status area
draw.line((0, status_h - 1, 1440, status_h - 1), fill=(200, 200, 200), width=1)

# Slight toolbar/search background strip (behind the search area, not drawing any icons/text)
search_y = 72
search_h = 191
strip_pad = 12
draw.rectangle((0, search_y, 1440, search_y + search_h + strip_pad), fill=(250, 250, 250))
draw.line((48, search_y + search_h + strip_pad - 6, 1392, search_y + search_h + strip_pad - 6),
          fill=(220, 220, 225), width=2)

# Filter strip background (subtle band behind filter pills)
filter_band_y = search_y + search_h + 6
filter_band_h = 86
draw.rounded_rectangle((48, filter_band_y, 1392, filter_band_y + filter_band_h),
                       radius=44, fill=(249, 250, 252))

# Separator under filters
sep_y = filter_band_y + filter_band_h + 18
draw.line((48, sep_y, 1392, sep_y), fill=(230, 230, 235), width=1)

# Card shadow and card backgrounds for listed events (rounded cards)
cards = [
    # (x1, y1, x2, y2) using detected card-ish positions from UI layout context
    (48, 525, 1392, 525 + 903),   # Buyers Breakfast card area
    (48, 1476, 1392, 1476 + 1108) # New York Tech Career Fair card area
]

for (x1, y1, x2, y2) in cards:
    # shadow
    shadow_offset = 10
    draw.rounded_rectangle(
        (x1 + shadow_offset, y1 + shadow_offset, x2 + shadow_offset, y2 + shadow_offset),
        radius=28, fill=(235, 235, 240)
    )
    # card background
    draw.rounded_rectangle((x1, y1, x2, y2), radius=22, fill=(255, 255, 255))

    # inside card: image/banner background band (kept as neutral darker band to imply image area)
    banner_h = int((y2 - y1) * 0.42)
    banner_radius = 18
    draw.rounded_rectangle((x1 + 16, y1 + 16, x2 - 16, y1 + 16 + banner_h),
                           radius=banner_radius, fill=(36, 40, 44))

    # subtle divider between banner and body of card
    draw.line((x1 + 24, y1 + 16 + banner_h + 12, x2 - 24, y1 + 16 + banner_h + 12),
              fill=(242, 242, 245), width=1)

    # light bottom separator for the card area
    draw.line((x1 + 8, y2 + 8, x2 - 8, y2 + 8), fill=(240, 240, 244), width=1)

# Lightweight separator between the two major card sections (in case additional spacing needed)
middle_sep_y = cards[0][3] + 32
draw.line((48, middle_sep_y, 1392, middle_sep_y), fill=(235, 235, 238), width=1)

# Bottom navigation bar background
nav_h = 120
nav_y = 2960 - nav_h
draw.rectangle((0, nav_y, 1440, 2960), fill=(255, 255, 255))
draw.line((0, nav_y, 1440, nav_y), fill=(226, 226, 230), width=1)

# Small top edge shadow for the entire content area (subtle)
draw.rectangle((0, sep_y + 6, 1440, sep_y + 8), fill=(245, 245, 247))

# Additional subtle horizontal guides to match app layout rhythm
for y in (search_y + search_h + 6, sep_y + 60, cards[0][3] + 18, cards[1][3] + 20):
    draw.line((48, y, 1392, y), fill=(245, 245, 247), width=1)

# --- Deterministic element pastes ---
# --- Auto-generated: paste detected elements at original positions ---

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/00_icon_Business.png
try:
    _c0 = get_crop(0, 249, 111)
    canvas.paste(_c0, (843, 406), _c0)
except Exception:
    pass
layout["Business"] = [843, 406, 1092, 517]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/01_icon_Anytime.png
try:
    _c1 = get_crop(1, 1344, 191)
    canvas.paste(_c1, (48, 72), _c1)
except Exception:
    pass
layout["Anytime"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/02_icon_1_Filter.png
try:
    _c2 = get_crop(2, 434, 144)
    canvas.paste(_c2, (0, 259), _c2)
except Exception:
    pass
layout["1_Filter"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/03_icon_Favorite_button.png
try:
    _c3 = get_crop(3, 144, 144)
    canvas.paste(_c3, (1092, 848), _c3)
except Exception:
    pass
layout["Favorite_button"] = [1092, 848, 1236, 992]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/04_icon_FAIR.png
try:
    _c4 = get_crop(4, 144, 144)
    canvas.paste(_c4, (1092, 1992), _c4)
except Exception:
    pass
layout["FAIR"] = [1092, 1992, 1236, 2136]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/05_icon_Overflow_menu_button.png
try:
    _c5 = get_crop(5, 144, 144)
    canvas.paste(_c5, (1236, 848), _c5)
except Exception:
    pass
layout["Overflow_menu_button"] = [1236, 848, 1380, 992]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/06_icon_FAIR.png
try:
    _c6 = get_crop(6, 144, 144)
    canvas.paste(_c6, (1236, 1992), _c6)
except Exception:
    pass
layout["FAIR"] = [1236, 1992, 1380, 2136]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/07_icon_9.39.png
try:
    _c7 = get_crop(7, 121, 111)
    canvas.paste(_c7, (57, 114), _c7)
except Exception:
    pass
layout["9.39"] = [57, 114, 178, 225]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/08_icon_icon_8.png
try:
    _c8 = get_crop(8, 56, 61)
    canvas.paste(_c8, (246, 1), _c8)
except Exception:
    pass
layout["icon_8"] = [246, 1, 302, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/09_icon_9.39.png
try:
    _c9 = get_crop(9, 54, 62)
    canvas.paste(_c9, (183, 0), _c9)
except Exception:
    pass
layout["9.39"] = [183, 0, 237, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/10_icon_icon_10.png
try:
    _c10 = get_crop(10, 60, 59)
    canvas.paste(_c10, (1318, 0), _c10)
except Exception:
    pass
layout["icon_10"] = [1318, 0, 1378, 59]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/11_icon_Search_forae.png
try:
    _c11 = get_crop(11, 60, 64)
    canvas.paste(_c11, (311, 0), _c11)
except Exception:
    pass
layout["Search_forae"] = [311, 0, 371, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/12_icon_icon_12.png
try:
    _c12 = get_crop(12, 64, 62)
    canvas.paste(_c12, (1209, 0), _c12)
except Exception:
    pass
layout["icon_12"] = [1209, 0, 1273, 62]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/13_icon_9.39.png
try:
    _c13 = get_crop(13, 58, 64)
    canvas.paste(_c13, (113, 0), _c13)
except Exception:
    pass
layout["9.39"] = [113, 0, 171, 64]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/14_icon_icon_14.png
try:
    _c14 = get_crop(14, 41, 61)
    canvas.paste(_c14, (1273, 0), _c14)
except Exception:
    pass
layout["icon_14"] = [1273, 0, 1314, 61]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/15_icon_New_York_Tech_Career_Fair_Exclusive_Tech.png
try:
    _c15 = get_crop(15, 1344, 1108)
    canvas.paste(_c15, (48, 1476), _c15)
except Exception:
    pass
layout["New_York_Tech_Career_Fair"] = [48, 1476, 1392, 2584]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/16_icon_Tech.png
try:
    _c16 = get_crop(16, 1344, 1108)
    canvas.paste(_c16, (48, 1476), _c16)
except Exception:
    pass
layout["Tech"] = [48, 1476, 1392, 2584]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/17_icon_ANNUAL_REENURY_CONFERENCE.png
try:
    _c17 = get_crop(17, 288, 156)
    canvas.paste(_c17, (864, 2804), _c17)
except Exception:
    pass
layout["ANNUAL_REENURY_CONFERENCE"] = [864, 2804, 1152, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/18_icon_Search_forae.png
try:
    _c18 = get_crop(18, 1344, 191)
    canvas.paste(_c18, (48, 72), _c18)
except Exception:
    pass
layout["Search_forae"] = [48, 72, 1392, 263]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/19_icon_Promoted.png
try:
    _c19 = get_crop(19, 43, 61)
    canvas.paste(_c19, (284, 1323), _c19)
except Exception:
    pass
layout["Promoted"] = [284, 1323, 327, 1384]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/20_icon_BUYER_BREAKFAST.png
try:
    _c20 = get_crop(20, 1344, 903)
    canvas.paste(_c20, (48, 525), _c20)
except Exception:
    pass
layout["BUYER_BREAKFAST"] = [48, 525, 1392, 1428]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/21_icon_New_York.png
try:
    _c21 = get_crop(21, 434, 144)
    canvas.paste(_c21, (0, 259), _c21)
except Exception:
    pass
layout["New_York"] = [0, 259, 434, 403]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/22_icon_EoeURAtION.png
try:
    _c22 = get_crop(22, 288, 156)
    canvas.paste(_c22, (288, 2804), _c22)
except Exception:
    pass
layout["EoeURAtION"] = [288, 2804, 576, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/23_icon_More.png
try:
    _c23 = get_crop(23, 288, 156)
    canvas.paste(_c23, (1152, 2804), _c23)
except Exception:
    pass
layout["More"] = [1152, 2804, 1440, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/24_icon_ANNUAL_REENURY_CONFERENCE.png
try:
    _c24 = get_crop(24, 288, 156)
    canvas.paste(_c24, (576, 2804), _c24)
except Exception:
    pass
layout["ANNUAL_REENURY_CONFERENCE"] = [576, 2804, 864, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/25_icon_Search_forae.png
try:
    _c25 = get_crop(25, 48, 62)
    canvas.paste(_c25, (383, 1), _c25)
except Exception:
    pass
layout["Search_forae"] = [383, 1, 431, 63]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/26_icon_NJ.png
try:
    _c26 = get_crop(26, 288, 156)
    canvas.paste(_c26, (0, 2804), _c26)
except Exception:
    pass
layout["NJ"] = [0, 2804, 288, 2960]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/27_icon_Moved_to_Virtual_Event.png
try:
    _c27 = get_crop(27, 423, 65)
    canvas.paste(_c27, (88, 2477), _c27)
except Exception:
    pass
layout["Moved_to_Virtual_Event"] = [88, 2477, 511, 2542]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/28_text_9.39.png
try:
    _c28 = get_crop(28, 94, 45)
    canvas.paste(_c28, (17, 15), _c28)
except Exception:
    pass
layout["9.39"] = [17, 15, 111, 60]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/29_text_Free.png
try:
    _c29 = get_crop(29, 77, 39)
    canvas.paste(_c29, (117, 1046), _c29)
except Exception:
    pass
layout["Free"] = [117, 1046, 194, 1085]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/30_text_Buyers_Breakfast.png
try:
    _c30 = get_crop(30, 470, 79)
    canvas.paste(_c30, (91, 1110), _c30)
except Exception:
    pass
layout["Buyers_Breakfast"] = [91, 1110, 561, 1189]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/31_text_Sat_Mar_23.png
try:
    _c31 = get_crop(31, 235, 53)
    canvas.paste(_c31, (90, 1196), _c31)
except Exception:
    pass
layout["Sat,_Mar_23"] = [90, 1196, 325, 1249]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/32_text_IO_O0_AM_EDT.png
try:
    _c32 = get_crop(32, 274, 45)
    canvas.paste(_c32, (346, 1197), _c32)
except Exception:
    pass
layout["IO:O0_AM_EDT"] = [346, 1197, 620, 1242]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/33_text_One_Pierrepont_Plaza_Cadman_Plaza_West_B.png
try:
    _c33 = get_crop(33, 1344, 903)
    canvas.paste(_c33, (48, 525), _c33)
except Exception:
    pass
layout["One_Pierrepont_Plaza,_Cad"] = [48, 525, 1392, 1428]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/34_text_Free.png
try:
    _c34 = get_crop(34, 80, 38)
    canvas.paste(_c34, (117, 2191), _c34)
except Exception:
    pass
layout["Free"] = [117, 2191, 197, 2229]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/35_text_Fri_Mar_29.png
try:
    _c35 = get_crop(35, 218, 51)
    canvas.paste(_c35, (91, 2419), _c35)
except Exception:
    pass
layout["Fri,_Mar_29"] = [91, 2419, 309, 2470]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/36_text_12_00_PM_EDT.png
try:
    _c36 = get_crop(36, 274, 45)
    canvas.paste(_c36, (330, 2420), _c36)
except Exception:
    pass
layout["12:00_PM_EDT"] = [330, 2420, 604, 2465]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/37_text_NJ.png
try:
    _c37 = get_crop(37, 85, 57)
    canvas.paste(_c37, (75, 2725), _c37)
except Exception:
    pass
layout["NJ"] = [75, 2725, 160, 2782]

# asset_path: data_engine/mock_unified_app_graph/extracted_assets/eventbrite/97cc452004d64d67bb4c5b51ffb8f6d4/step_06_2024_3_20_17_38_97cc452004d64d67bb4c5b51ffb8f6d4-8/38_text_ANNUAL_REENURY_CONFERENCE.png
try:
    _c38 = get_crop(38, 288, 156)
    canvas.paste(_c38, (576, 2804), _c38)
except Exception:
    pass
layout["ANNUAL_REENURY_CONFERENCE"] = [576, 2804, 864, 2960]
